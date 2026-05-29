import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from dataset import pil_to_tensor
from model import PromptIR


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


def tensor_to_uint8_chw(tensor):
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
    return arr


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="./release_folder/hw4_realse_dataset",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./hw4_checkpoints_promptir_l1grad/best_fullval_promptir.pth",
    )
    parser.add_argument("--output_npz", type=str, default="./pred.npz")
    parser.add_argument("--output_zip", type=str, default="./submission.zip")

    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--use_ema", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    test_dir = Path(args.data_root) / "test" / "degraded"

    assert test_dir.exists(), f"Missing test directory: {test_dir}"
    assert Path(args.checkpoint).exists(), f"Missing checkpoint: {args.checkpoint}"

    model = PromptIR(dim=args.dim).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    if args.use_ema and "ema_state_dict" in checkpoint:
        print("Loading EMA weights.")
        model.load_state_dict(checkpoint["ema_state_dict"])
    else:
        print("Loading normal model weights.")
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    image_paths = sorted(
        test_dir.glob("*.png"),
        key=lambda p: int(p.stem),
    )

    print(f"Number of test images: {len(image_paths)}")

    results = {}

    for idx, image_path in enumerate(image_paths):
        img = Image.open(image_path).convert("RGB")
        image = pil_to_tensor(img)

        pred = tile_restore(
            model=model,
            image=image,
            device=device,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )

        pred_uint8 = tensor_to_uint8_chw(pred)

        results[image_path.name] = pred_uint8

        print(
            f"[{idx + 1}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"shape={pred_uint8.shape} | "
            f"dtype={pred_uint8.dtype}"
        )

    np.savez(args.output_npz, **results)
    print(f"Saved npz to: {args.output_npz}")

    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(args.output_npz, arcname="pred.npz")

    print(f"Saved submission zip to: {args.output_zip}")
    print("Zip should contain pred.npz.")


if __name__ == "__main__":
    main()
