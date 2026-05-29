import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def pil_to_tensor(img):
    """Convert PIL image to CHW float tensor in range [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    arr = arr[:, :, :3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def build_train_val_pairs(data_root, train_per_type=1500, val_per_type=100):
    """
    Build train/validation pairs.

    Expected:
        train/degraded/rain-1.png ... rain-1600.png
        train/degraded/snow-1.png ... snow-1600.png
        train/clean/rain_clean-1.png ... rain_clean-1600.png
        train/clean/snow_clean-1.png ... snow_clean-1600.png

    Split:
        rain/snow 1-1500 -> train
        rain/snow 1501-1600 -> validation
    """
    data_root = Path(data_root)

    degraded_dir = data_root / "train" / "degraded"
    clean_dir = data_root / "train" / "clean"

    print(f"[Dataset] data_root: {data_root}")
    print(f"[Dataset] degraded_dir exists: {degraded_dir.exists()}")
    print(f"[Dataset] clean_dir exists: {clean_dir.exists()}")

    if not degraded_dir.exists() or not clean_dir.exists():
        raise FileNotFoundError(
            f"Cannot find train/degraded or train/clean under {data_root}"
        )

    train_pairs = []
    val_pairs = []

    for deg_type in ["rain", "snow"]:
        pairs = []

        for idx in range(1, 1601):
            degraded_path = degraded_dir / f"{deg_type}-{idx}.png"
            clean_path = clean_dir / f"{deg_type}_clean-{idx}.png"

            if degraded_path.exists() and clean_path.exists():
                pairs.append({
                    "degraded": degraded_path,
                    "clean": clean_path,
                    "type": deg_type,
                    "index": idx,
                })

        pairs = sorted(pairs, key=lambda x: x["index"])
        print(f"[Dataset] Found {len(pairs)} pairs for {deg_type}")

        train_pairs.extend(pairs[:train_per_type])
        val_pairs.extend(pairs[train_per_type:train_per_type + val_per_type])

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    print(f"[Dataset] Total train pairs: {len(train_pairs)}")
    print(f"[Dataset] Total val pairs: {len(val_pairs)}")

    if len(train_pairs) == 0:
        raise RuntimeError("No training pairs found. Please check data_root.")

    return train_pairs, val_pairs


class RestorationRandomCropDataset(Dataset):
    """
    Random-crop training dataset with fixed epoch size.

    This makes each epoch contain a fixed number of random crops.
    For example:
        steps_per_epoch = 1076
        batch_size = 4
        epoch_size = 4304

    It only samples from the provided training images.
    """

    def __init__(self, pairs, patch_size=256, epoch_size=4304):
        self.pairs = pairs
        self.patch_size = patch_size
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def resize_if_needed(self, degraded, clean):
        _, h, w = degraded.shape
        ps = self.patch_size

        if h < ps or w < ps:
            new_h = max(h, ps)
            new_w = max(w, ps)

            degraded = F.interpolate(
                degraded.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            clean = F.interpolate(
                clean.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return degraded, clean

    def random_crop(self, degraded, clean):
        degraded, clean = self.resize_if_needed(degraded, clean)

        _, h, w = degraded.shape
        ps = self.patch_size

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)

        degraded = degraded[:, top:top + ps, left:left + ps]
        clean = clean[:, top:top + ps, left:left + ps]

        return degraded, clean

    def augment(self, degraded, clean):
        if random.random() < 0.5:
            degraded = torch.flip(degraded, dims=[2])
            clean = torch.flip(clean, dims=[2])

        if random.random() < 0.5:
            degraded = torch.flip(degraded, dims=[1])
            clean = torch.flip(clean, dims=[1])

        k = random.randint(0, 3)
        if k > 0:
            degraded = torch.rot90(degraded, k, dims=[1, 2])
            clean = torch.rot90(clean, k, dims=[1, 2])

        return degraded, clean

    def __getitem__(self, idx):
        item = random.choice(self.pairs)

        degraded_img = Image.open(item["degraded"]).convert("RGB")
        clean_img = Image.open(item["clean"]).convert("RGB")

        degraded = pil_to_tensor(degraded_img)
        clean = pil_to_tensor(clean_img)

        degraded, clean = self.random_crop(degraded, clean)
        degraded, clean = self.augment(degraded, clean)

        return degraded, clean


class RestorationFullValDataset(Dataset):
    """
    Full-image validation dataset.

    Important:
    This dataset returns whole validation images.
    No center crop. No random crop.
    """

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        degraded_img = Image.open(item["degraded"]).convert("RGB")
        clean_img = Image.open(item["clean"]).convert("RGB")

        degraded = pil_to_tensor(degraded_img)
        clean = pil_to_tensor(clean_img)

        return degraded, clean, item["type"], item["index"]
