import argparse
import copy
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import (
    RestorationFullValDataset,
    RestorationRandomCropDataset,
    build_train_val_pairs,
)
from model import PromptIR


class GradientLoss(nn.Module):
    """Sobel gradient loss for preserving image structures."""

    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient(self, x):
        channels = x.shape[1]

        sobel_x = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sobel_y = self.sobel_y.to(device=x.device, dtype=x.dtype)

        sobel_x = sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(channels, 1, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        pred_grad = self.gradient(pred)
        target_grad = self.gradient(target)

        return F.l1_loss(pred_grad, target_grad)


class RestorationLoss(nn.Module):
    """
    Restoration objective:
        total_loss = L1(pred, target) + grad_weight * GradientLoss(pred, target)
    """

    def __init__(self, grad_weight=0.05):
        super().__init__()
        self.grad_weight = grad_weight
        self.l1 = nn.L1Loss()
        self.grad = GradientLoss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        grad_loss = self.grad(pred, target)
        total_loss = l1_loss + self.grad_weight * grad_loss

        return total_loss, l1_loss, grad_loss


class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for key in ema_state.keys():
            ema_state[key].mul_(self.decay).add_(
                model_state[key].detach(),
                alpha=1.0 - self.decay,
            )


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False

    return distributed, rank, world_size, local_rank


def cleanup_distributed(distributed):
    if distributed:
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed, rank=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calc_psnr(pred, target):
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    mse = F.mse_loss(pred, target, reduction="mean")

    if mse.item() == 0:
        return 100.0

    return (-10.0 * torch.log10(mse)).item()


def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup followed by cosine decay."""

    def lr_lambda(epoch_idx):
        current_epoch = epoch_idx + 1

        if current_epoch <= warmup_epochs:
            return current_epoch / warmup_epochs

        progress = (current_epoch - warmup_epochs) / max(
            1,
            total_epochs - warmup_epochs,
        )

        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def tile_restore(model, image, device, tile_size=256, overlap=64):
    model.eval()

    _, h, w = image.shape

    if h <= tile_size and w <= tile_size:
        pred = model(image.unsqueeze(0).to(device)).squeeze(0).cpu()
        return pred.clamp(0.0, 1.0)

    stride = tile_size - overlap

    output = torch.zeros_like(image)
    weight = torch.zeros_like(image)

    for top_origin in range(0, h, stride):
        for left_origin in range(0, w, stride):
            bottom = min(top_origin + tile_size, h)
            right = min(left_origin + tile_size, w)

            top = max(bottom - tile_size, 0)
            left = max(right - tile_size, 0)

            patch = image[:, top:bottom, left:right]

            pad_h = tile_size - patch.shape[1]
            pad_w = tile_size - patch.shape[2]

            if pad_h > 0 or pad_w > 0:
                patch = F.pad(
                    patch,
                    pad=(0, pad_w, 0, pad_h),
                    mode="reflect",
                )

            pred_patch = model(patch.unsqueeze(0).to(device))
            pred_patch = pred_patch.squeeze(0).cpu()
            pred_patch = pred_patch[:, :bottom - top, :right - left]

            output[:, top:bottom, left:right] += pred_patch
            weight[:, top:bottom, left:right] += 1.0

    output = output / torch.clamp(weight, min=1.0)
    return output.clamp(0.0, 1.0)


def get_raw_model(model):
    if isinstance(model, DDP):
        return model.module
    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    epochs,
    ema,
    rank,
):
    model.train()

    running_loss = 0.0
    running_l1 = 0.0
    running_grad = 0.0
    running_mse = 0.0
    running_psnr = 0.0

    for step, (degraded, clean) in enumerate(loader):
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(degraded)
            loss, l1_loss, grad_loss = criterion(pred, clean)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(get_raw_model(model))

        with torch.no_grad():
            mse = F.mse_loss(
                torch.clamp(pred, 0.0, 1.0),
                torch.clamp(clean, 0.0, 1.0),
            )
            batch_psnr = calc_psnr(pred.detach(), clean)

        running_loss += loss.item()
        running_l1 += l1_loss.item()
        running_grad += grad_loss.item()
        running_mse += mse.item()
        running_psnr += batch_psnr

        if is_main_process(rank) and (step + 1) % 100 == 0:
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Step [{step + 1}/{len(loader)}] "
                f"Loss: {running_loss / (step + 1):.4f} "
                f"L1: {running_l1 / (step + 1):.4f} "
                f"MSE: {running_mse / (step + 1):.6f} "
                f"Grad: {running_grad / (step + 1):.4f} "
                f"Patch PSNR: {running_psnr / (step + 1):.2f}"
            )

    n = len(loader)

    return {
        "loss": running_loss / n,
        "l1": running_l1 / n,
        "mse": running_mse / n,
        "grad": running_grad / n,
        "patch_psnr": running_psnr / n,
    }


@torch.no_grad()
def validate_full_image(model, loader, device, tile_size, overlap):
    model.eval()

    psnr_list = []
    rain_list = []
    snow_list = []

    for idx, (degraded, clean, deg_type, img_index) in enumerate(loader):
        degraded = degraded.squeeze(0)
        clean = clean.squeeze(0)

        pred = tile_restore(
            model=model,
            image=degraded,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
        )

        psnr = calc_psnr(pred, clean)
        psnr_list.append(psnr)

        deg_type_str = deg_type[0]

        if deg_type_str == "rain":
            rain_list.append(psnr)
        else:
            snow_list.append(psnr)

        print(
            f"[Full Val {idx + 1}/{len(loader)}] "
            f"{deg_type_str}-{int(img_index.item())} "
            f"PSNR: {psnr:.2f}"
        )

    full_psnr = float(np.mean(psnr_list))
    rain_psnr = float(np.mean(rain_list)) if rain_list else 0.0
    snow_psnr = float(np.mean(snow_list)) if snow_list else 0.0

    return full_psnr, rain_psnr, snow_psnr


def save_history(history, path):
    keys = [
        "epoch",
        "train_loss",
        "train_l1",
        "train_mse",
        "train_grad",
        "train_patch_psnr",
        "val_psnr",
        "val_psnr_rain",
        "val_psnr_snow",
        "lr",
        "steps",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for i in range(len(history["epoch"])):
            row = {key: history[key][i] for key in keys}
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="./release_folder/hw4_realse_dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./hw4_checkpoints_promptir_l1grad",
    )

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup_epochs", type=int, default=15)

    # In DDP, batch_size means per-GPU batch size.
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=256)

    # Number of optimizer update steps per epoch.
    parser.add_argument("--steps_per_epoch", type=int, default=1076)

    parser.add_argument("--dim", type=int, default=48)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_weight", type=float, default=0.05)

    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)

    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume_model_only", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()

    set_seed(args.seed, rank=rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process(rank):
        print("Device:", device)
        print("Distributed:", distributed)
        print("World size:", world_size)
        print("Local rank:", local_rank)

    save_dir = Path(args.save_dir)

    if is_main_process(rank):
        save_dir.mkdir(parents=True, exist_ok=True)

    if distributed:
        dist.barrier()

    best_path = save_dir / "best_fullval_promptir.pth"
    last_path = save_dir / "last_promptir.pth"
    history_path = save_dir / "history.csv"

    train_pairs, val_pairs = build_train_val_pairs(args.data_root)

    global_batch_size = args.batch_size * world_size
    epoch_size = args.steps_per_epoch * global_batch_size

    train_dataset = RestorationRandomCropDataset(
        train_pairs,
        patch_size=args.patch_size,
        epoch_size=epoch_size,
    )

    val_dataset = RestorationFullValDataset(val_pairs)

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = None

    if is_main_process(rank):
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )

        print(f"Per-GPU batch size: {args.batch_size}")
        print(f"Global batch size: {global_batch_size}")
        print(f"Train steps per epoch: {len(train_loader)}")
        print(f"Full validation images: {len(val_loader)}")

    model = PromptIR(dim=args.dim).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if is_main_process(rank):
        print(f"Trainable parameters: {num_params / 1e6:.2f} M")

    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    criterion = RestorationLoss(grad_weight=args.grad_weight).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    best_full_val = -1.0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_l1": [],
        "train_mse": [],
        "train_grad": [],
        "train_patch_psnr": [],
        "val_psnr": [],
        "val_psnr_rain": [],
        "val_psnr_snow": [],
        "lr": [],
        "steps": [],
    }

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if args.use_ema and ema is not None and "ema_state_dict" in checkpoint:
            ema.ema.load_state_dict(checkpoint["ema_state_dict"])

        if not args.resume_model_only:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint.get("epoch", 0) + 1
            best_full_val = checkpoint.get("best_full_val_psnr", -1.0)

            if "history" in checkpoint:
                history = checkpoint["history"]

            if is_main_process(rank):
                print(f"Resumed full checkpoint: {args.resume}")
                print(f"Start epoch: {start_epoch}")
                print(f"Best full-val PSNR so far: {best_full_val:.2f}")
        else:
            if is_main_process(rank):
                print(f"Loaded model weights only: {args.resume}")

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            ema=ema,
            rank=rank,
        )

        if is_main_process(rank):
            eval_model = ema.ema if ema is not None else get_raw_model(model)

            val_psnr, val_psnr_rain, val_psnr_snow = validate_full_image(
                model=eval_model,
                loader=val_loader,
                device=device,
                tile_size=args.tile_size,
                overlap=args.overlap,
            )
        else:
            val_psnr = 0.0
            val_psnr_rain = 0.0
            val_psnr_snow = 0.0

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if is_main_process(rank):
            print(
                f"Epoch {epoch} Summary | "
                f"Train Loss: {train_stats['loss']:.4f}, "
                f"Train L1: {train_stats['l1']:.4f}, "
                f"Train MSE: {train_stats['mse']:.6f}, "
                f"Train Grad: {train_stats['grad']:.4f}, "
                f"Train Patch PSNR: {train_stats['patch_psnr']:.2f}, "
                f"Full Val PSNR: {val_psnr:.2f}, "
                f"Rain Full Val PSNR: {val_psnr_rain:.2f}, "
                f"Snow Full Val PSNR: {val_psnr_snow:.2f}, "
                f"LR: {current_lr:.8f}"
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_stats["loss"])
            history["train_l1"].append(train_stats["l1"])
            history["train_mse"].append(train_stats["mse"])
            history["train_grad"].append(train_stats["grad"])
            history["train_patch_psnr"].append(train_stats["patch_psnr"])
            history["val_psnr"].append(val_psnr)
            history["val_psnr_rain"].append(val_psnr_rain)
            history["val_psnr_snow"].append(val_psnr_snow)
            history["lr"].append(current_lr)
            history["steps"].append(len(train_loader))

            raw_model = get_raw_model(model)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_full_val_psnr": best_full_val,
                "history": history,
                "config": vars(args),
            }

            if ema is not None:
                checkpoint["ema_state_dict"] = ema.ema.state_dict()

            torch.save(checkpoint, last_path)

            if val_psnr > best_full_val:
                best_full_val = val_psnr
                checkpoint["best_full_val_psnr"] = best_full_val
                torch.save(checkpoint, best_path)

                print(f"New best full-val checkpoint saved: {best_path}")
                print(f"Best Full Val PSNR: {best_full_val:.2f}")

            save_history(history, history_path)

        if distributed:
            dist.barrier()

    if is_main_process(rank):
        print("Training finished.")
        print(f"Best Full Val PSNR: {best_full_val:.2f}")
        print(f"Best checkpoint: {best_path}")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
