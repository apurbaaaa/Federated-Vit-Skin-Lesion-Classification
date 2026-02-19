"""
Data pipeline — ISIC 2019 (8-class, no UNK).

Features
────────
  • Patient-level split  (StratifiedGroupKFold)
  • Strong train augment (RandomResizedCrop 384, HFlip, VFlip, Rotation ±30,
                          ColorJitter, RandAugment n=3 m=12)
  • Eval: Resize → CenterCrop(384)
  • Metadata one-hot encoding  (age + sex_oh + site_oh → 13-dim)
  • Precomputed segmentation mask  → 4-channel input
  • 8-transform deterministic TTA dataset
  • WeightedRandomSampler
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF

try:
    from torchvision.transforms import RandAugment
    HAS_RANDAUGMENT = True
except ImportError:
    HAS_RANDAUGMENT = False


# ============================================================================
# Constants
# ============================================================================

VALID_CLASSES: List[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES: int = 8

LABEL_TO_IDX: Dict[str, int] = {n: i for i, n in enumerate(VALID_CLASSES)}
IDX_TO_LABEL: Dict[int, str] = {i: n for i, n in enumerate(VALID_CLASSES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

SEX_VOCAB: Dict[str, int] = {"male": 0, "female": 1, "unknown": 2}
NUM_SEX = len(SEX_VOCAB)

SITE_VOCAB: Dict[str, int] = {
    "anterior torso": 0, "upper extremity": 1, "lower extremity": 2,
    "posterior torso": 3, "lateral torso": 4, "head/neck": 5,
    "palms/soles": 6, "oral/genital": 7, "unknown": 8,
}
NUM_SITE = len(SITE_VOCAB)

META_DIM = 1 + NUM_SEX + NUM_SITE  # 13


# ============================================================================
# Metadata helpers
# ============================================================================

def encode_metadata_vector(age_norm: float, sex_idx: int, site_idx: int) -> torch.Tensor:
    """Return (META_DIM,) = [age, sex_onehot(3), site_onehot(9)]."""
    vec = torch.zeros(META_DIM, dtype=torch.float32)
    vec[0] = age_norm
    vec[1 + sex_idx] = 1.0
    vec[1 + NUM_SEX + site_idx] = 1.0
    return vec


# ============================================================================
# Transforms — Train
# ============================================================================

class TrainTransform:
    def __init__(self, image_size: int = 384, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.image_size = image_size
        self.scale = tuple(cfg.get("random_resized_crop", {}).get("scale", [0.7, 1.0]))
        self.ratio = tuple(cfg.get("random_resized_crop", {}).get("ratio", [0.9, 1.1]))
        self.hflip = cfg.get("horizontal_flip", True)
        self.vflip = cfg.get("vertical_flip", True)
        self.rotation = cfg.get("rotation", 30)

        cj = cfg.get("color_jitter", {})
        if cj:
            self.color_jitter = transforms.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.2),
                hue=cj.get("hue", 0.0),
            )
        else:
            self.color_jitter = None

        ra = cfg.get("randaugment", {})
        if ra.get("enabled", True) and HAS_RANDAUGMENT:
            self.randaugment = RandAugment(num_ops=ra.get("n", 3), magnitude=ra.get("m", 12))
        else:
            self.randaugment = None

    def __call__(self, image: Image.Image, mask: Optional[Image.Image] = None):
        # RandomResizedCrop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=self.scale, ratio=self.ratio)
        image = TF.resized_crop(image, i, j, h, w, (self.image_size, self.image_size),
                                interpolation=TF.InterpolationMode.BICUBIC)
        if mask is not None:
            mask = TF.resized_crop(mask, i, j, h, w, (self.image_size, self.image_size),
                                   interpolation=TF.InterpolationMode.NEAREST)

        # HFlip
        if self.hflip and torch.rand(1) < 0.5:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        # VFlip
        if self.vflip and torch.rand(1) < 0.5:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)

        # Rotation
        if self.rotation > 0:
            angle = float(torch.empty(1).uniform_(-self.rotation, self.rotation))
            image = TF.rotate(image, angle)
            if mask is not None:
                mask = TF.rotate(mask, angle)

        # ColorJitter (image only)
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        # RandAugment (image only)
        if self.randaugment is not None:
            image = self.randaugment(image)

        return self._to_tensor(image, mask)

    def _to_tensor(self, image, mask):
        img_t = TF.to_tensor(image)
        img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
        mask_t = None
        if mask is not None:
            mask_t = TF.to_tensor(mask)
            mask_t = (mask_t - 0.5) / 0.5
        return img_t, mask_t


# ============================================================================
# Transforms — Eval  (Resize + CenterCrop)
# ============================================================================

class EvalTransform:
    def __init__(self, image_size: int = 384):
        self.image_size = image_size
        self.resize_size = int(image_size * 1.14)

    def __call__(self, image: Image.Image, mask: Optional[Image.Image] = None):
        image = TF.resize(image, self.resize_size, interpolation=TF.InterpolationMode.BICUBIC)
        image = TF.center_crop(image, self.image_size)
        if mask is not None:
            mask = TF.resize(mask, self.resize_size, interpolation=TF.InterpolationMode.NEAREST)
            mask = TF.center_crop(mask, self.image_size)
        return self._to_tensor(image, mask)

    def _to_tensor(self, image, mask):
        img_t = TF.to_tensor(image)
        img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
        mask_t = None
        if mask is not None:
            mask_t = TF.to_tensor(mask)
            mask_t = (mask_t - 0.5) / 0.5
        return img_t, mask_t


# ============================================================================
# Dataset
# ============================================================================

class ISICDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 384,
        is_train: bool = True,
        use_metadata: bool = True,
        use_segmentation_mask: bool = True,
        mask_dir: Optional[str] = None,
        aug_cfg: Optional[dict] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.use_metadata = use_metadata
        self.use_seg = use_segmentation_mask
        self.mask_dir = Path(mask_dir) if mask_dir else None
        if is_train:
            self.transform = TrainTransform(image_size, cfg=aug_cfg)
        else:
            self.transform = EvalTransform(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        mask = None
        if self.use_seg and self.mask_dir is not None:
            mp = self.mask_dir / f"{row['image']}.png"
            if mp.exists():
                mask = Image.open(mp).convert("L")

        img_t, mask_t = self.transform(image, mask)
        if self.use_seg and mask_t is not None:
            img_t = torch.cat([img_t, mask_t], dim=0)  # 4-ch

        out: Dict = {"image": img_t, "label": int(row["label"])}

        if self.use_metadata:
            age  = float(row.get("age_norm", 0.0))
            sex  = int(row.get("sex_idx", SEX_VOCAB["unknown"]))
            site = int(row.get("site_idx", SITE_VOCAB["unknown"]))
            out["metadata"] = encode_metadata_vector(age, sex, site)

        return out


# ============================================================================
# TTA Dataset — 8 deterministic transforms
# ============================================================================

_TTA_TRANSFORMS: List[str] = [
    "original", "hflip", "vflip",
    "rot90", "rot180", "rot270",
    "bright_up", "bright_down",
]


class TTADataset(Dataset):
    """Produces (num_tta, C, H, W) per sample using 8 deterministic augments.

    Transforms:
        0  original
        1  HFlip
        2  VFlip
        3  Rotate 90
        4  Rotate 180
        5  Rotate 270
        6  Brightness up   (×1.15)
        7  Brightness down (×0.85)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 384,
        use_metadata: bool = True,
        use_segmentation_mask: bool = True,
        mask_dir: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.use_metadata = use_metadata
        self.use_seg = use_segmentation_mask
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.eval_tf = EvalTransform(image_size)

    def __len__(self) -> int:
        return len(self.df)

    # ---- internal helpers ------------------------------------------------
    @staticmethod
    def _apply_tta(image: Image.Image, mask: Optional[Image.Image], name: str):
        if name == "hflip":
            image = TF.hflip(image)
            if mask: mask = TF.hflip(mask)
        elif name == "vflip":
            image = TF.vflip(image)
            if mask: mask = TF.vflip(mask)
        elif name == "rot90":
            image = TF.rotate(image, 90)
            if mask: mask = TF.rotate(mask, 90)
        elif name == "rot180":
            image = TF.rotate(image, 180)
            if mask: mask = TF.rotate(mask, 180)
        elif name == "rot270":
            image = TF.rotate(image, 270)
            if mask: mask = TF.rotate(mask, 270)
        elif name == "bright_up":
            image = ImageEnhance.Brightness(image).enhance(1.15)
        elif name == "bright_down":
            image = ImageEnhance.Brightness(image).enhance(0.85)
        # "original" → no-op
        return image, mask

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image_orig = Image.open(row["image_path"]).convert("RGB")

        mask_orig: Optional[Image.Image] = None
        if self.use_seg and self.mask_dir is not None:
            mp = self.mask_dir / f"{row['image']}.png"
            if mp.exists():
                mask_orig = Image.open(mp).convert("L")

        tensors = []
        for tname in _TTA_TRANSFORMS:
            img_copy = image_orig.copy()
            msk_copy = mask_orig.copy() if mask_orig else None
            img_copy, msk_copy = self._apply_tta(img_copy, msk_copy, tname)
            img_t, mask_t = self.eval_tf(img_copy, msk_copy)
            if self.use_seg and mask_t is not None:
                img_t = torch.cat([img_t, mask_t], dim=0)
            tensors.append(img_t.unsqueeze(0))

        out: Dict = {
            "images": torch.cat(tensors, dim=0),  # (8, C, H, W)
            "label": int(row["label"]),
        }

        if self.use_metadata:
            age  = float(row.get("age_norm", 0.0))
            sex  = int(row.get("sex_idx", SEX_VOCAB["unknown"]))
            site = int(row.get("site_idx", SITE_VOCAB["unknown"]))
            out["metadata"] = encode_metadata_vector(age, sex, site)

        return out


# ============================================================================
# CSV / metadata loaders
# ============================================================================

def _parse_groundtruth_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    all_cls = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    cols = [c for c in all_cls if c in df.columns]
    df["dx"] = df[cols].idxmax(axis=1)
    df = df[df["dx"].isin(VALID_CLASSES)].copy()
    df["label"] = df["dx"].map(LABEL_TO_IDX)
    return df[["image", "dx", "label"]]


def _load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "lesion_id" not in df.columns:
        df["lesion_id"] = df["image"]
    # age
    age_col = "age_approx" if "age_approx" in df.columns else None
    df["age_norm"] = (df[age_col].fillna(50).clip(0, 90) / 90.0) if age_col else 0.5
    # sex
    sex_col = "sex" if "sex" in df.columns else None
    df["sex_idx"] = (
        df[sex_col].fillna("unknown").str.lower()
        .map(lambda x: SEX_VOCAB.get(x, SEX_VOCAB["unknown"]))
        if sex_col else SEX_VOCAB["unknown"]
    )
    # anatomical site — ISIC 2019 uses either column name
    for col in ("anatom_site_general", "anatom_site_general_challenge"):
        if col in df.columns:
            df["site_idx"] = df[col].fillna("unknown").str.lower().map(
                lambda x: SITE_VOCAB.get(x, SITE_VOCAB["unknown"]))
            break
    else:
        df["site_idx"] = SITE_VOCAB["unknown"]
    return df


def load_isic_data(isic_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full_train_df, test_df) — splitting is done in the CV loop."""
    isic = Path(isic_dir)

    # --- Training (required) ------------------------------------------------
    train_gt   = _parse_groundtruth_csv(isic / "ISIC_2019_Training_GroundTruth.csv")
    train_meta = _load_metadata(isic / "ISIC_2019_Training_Metadata.csv")
    train_df   = train_gt.merge(train_meta, on="image", how="left")
    train_img  = isic / "ISIC_2019_Training_Input"
    train_df["image_path"] = train_df["image"].apply(lambda x: str(train_img / f"{x}.jpg"))

    # Drop rows whose image file truly doesn't exist (guards against CSV/dir mismatch)
    missing = ~train_df["image_path"].apply(lambda p: Path(p).exists())
    if missing.any():
        print(f"[Data] WARNING: {missing.sum():,} training images not found on disk — dropping.")
        train_df = train_df[~missing].reset_index(drop=True)

    print(f"[Data] Training samples (8-class): {len(train_df):,}")

    # --- Test (optional — may not be distributed) ---------------------------
    test_gt_path   = isic / "ISIC_2019_Test_GroundTruth.csv"
    test_meta_path = isic / "ISIC_2019_Test_Metadata.csv"
    test_img       = isic / "ISIC_2019_Test_Input"

    if test_gt_path.exists() and test_meta_path.exists():
        test_gt   = _parse_groundtruth_csv(test_gt_path)
        test_meta = _load_metadata(test_meta_path)
        test_df   = test_gt.merge(test_meta, on="image", how="left")
        test_df["image_path"] = test_df["image"].apply(lambda x: str(test_img / f"{x}.jpg"))
        missing_t = ~test_df["image_path"].apply(lambda p: Path(p).exists())
        if missing_t.any():
            print(f"[Data] WARNING: {missing_t.sum():,} test images not found — dropping.")
            test_df = test_df[~missing_t].reset_index(drop=True)
        print(f"[Data] Test samples     (8-class): {len(test_df):,}")
    elif test_meta_path.exists() and test_img.exists():
        # No groundtruth but we have images — build unlabelled test df
        print("[Data] No test GT found — building unlabelled test set for inference only.")
        test_meta = _load_metadata(test_meta_path)
        test_df = test_meta[["image"]].copy()
        test_df["dx"]    = "MEL"   # placeholder
        test_df["label"] = 0       # placeholder
        test_df = test_df.merge(test_meta, on="image", how="left")
        test_df["image_path"] = test_df["image"].apply(lambda x: str(test_img / f"{x}.jpg"))
        test_df = test_df[test_df["image_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        print(f"[Data] Unlabelled test images: {len(test_df):,}")
    else:
        print("[Data] No test data found — test evaluation will be skipped.")
        test_df = pd.DataFrame(columns=train_df.columns)

    return train_df, test_df


# ============================================================================
# DataLoader builder
# ============================================================================

def build_fold_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
) -> Tuple[DataLoader, DataLoader]:
    """Build train + val loaders for one CV fold."""
    d  = config.get("data", {})
    m  = config.get("model", {})
    t  = config.get("training", {})
    ag = config.get("augmentation", {}).get("train", {})

    img_sz = m.get("image_size", 384)
    bs     = t.get("batch_size", 16)
    nw     = t.get("num_workers", 8)
    pm     = t.get("pin_memory", True)
    use_meta = m.get("metadata", {}).get("enabled", True)
    use_seg  = d.get("use_segmentation_mask", True)
    mask_dir = d.get("segmentation_mask_dir", "./masks")

    train_ds = ISICDataset(train_df, img_sz, True,  use_meta, use_seg, mask_dir, aug_cfg=ag)
    val_ds   = ISICDataset(val_df,   img_sz, False, use_meta, use_seg, mask_dir)

    # WeightedRandomSampler
    sampler = None
    if t.get("use_weighted_sampler", True):
        labels = train_df["label"].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
        counts[counts == 0] = 1.0
        w = 1.0 / counts
        sample_w = w[labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler,
        num_workers=nw, pin_memory=pm, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pm,
    )
    return train_loader, val_loader


def build_tta_loader(
    df: pd.DataFrame,
    config: dict,
) -> DataLoader:
    """Build TTA dataloader for test set."""
    d = config.get("data", {})
    m = config.get("model", {})
    t = config.get("training", {})
    ds = TTADataset(
        df, m.get("image_size", 384),
        use_metadata=m.get("metadata", {}).get("enabled", True),
        use_segmentation_mask=d.get("use_segmentation_mask", True),
        mask_dir=d.get("segmentation_mask_dir", "./masks"),
    )
    return DataLoader(
        ds, batch_size=max(1, t.get("batch_size", 16) // 4),
        shuffle=False, num_workers=t.get("num_workers", 8),
        pin_memory=t.get("pin_memory", True),
    )


def build_test_loader(
    df: pd.DataFrame,
    config: dict,
) -> DataLoader:
    """Non-TTA test loader."""
    d = config.get("data", {})
    m = config.get("model", {})
    t = config.get("training", {})
    ds = ISICDataset(
        df, m.get("image_size", 384), False,
        use_metadata=m.get("metadata", {}).get("enabled", True),
        use_segmentation_mask=d.get("use_segmentation_mask", True),
        mask_dir=d.get("segmentation_mask_dir", "./masks"),
    )
    return DataLoader(
        ds, batch_size=t.get("batch_size", 16), shuffle=False,
        num_workers=t.get("num_workers", 8), pin_memory=t.get("pin_memory", True),
    )


def print_class_distribution(df: pd.DataFrame, name: str) -> None:
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    print(f"\n[{name}] Class Distribution ({total:,} samples):")
    for idx in range(NUM_CLASSES):
        c = counts.get(idx, 0)
        print(f"  {idx} {VALID_CLASSES[idx]:5s}: {c:6,} ({100*c/total:5.2f}%)")
