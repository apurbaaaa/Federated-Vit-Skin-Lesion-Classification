"""
Data utilities for ISIC 2019 skin lesion classification (8-class, no UNK).

Features:
    - Patient-level split (StratifiedGroupKFold)
    - UNK class removed before splitting
    - Metadata loading and encoding
    - Optional segmentation mask loading
    - SOTA augmentation (RandAugment, ColorJitter, etc.)
    - K-fold support
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import DataLoader, Dataset
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

LABEL_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(VALID_CLASSES)}
IDX_TO_LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(VALID_CLASSES)}

IMAGENET_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

# Metadata vocabulary
SEX_VOCAB: Dict[str, int] = {"male": 0, "female": 1, "unknown": 2}
SITE_VOCAB: Dict[str, int] = {
    "anterior torso": 0,
    "upper extremity": 1,
    "lower extremity": 2,
    "posterior torso": 3,
    "lateral torso": 4,
    "head/neck": 5,
    "palms/soles": 6,
    "oral/genital": 7,
    "unknown": 8,
}


# ============================================================================
# Transforms
# ============================================================================

def get_train_transforms(
    image_size: int = 384,
    use_randaugment: bool = True,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    crop_scale: Tuple[float, float] = (0.7, 1.0),
    crop_ratio: Tuple[float, float] = (0.9, 1.1),
    color_jitter: bool = True,
    rotation: int = 30,
) -> transforms.Compose:
    """Strong training augmentations."""
    transform_list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=crop_scale,
            ratio=crop_ratio,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(rotation),
    ]
    
    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            )
        )
    
    if use_randaugment and HAS_RANDAUGMENT:
        transform_list.append(
            RandAugment(num_ops=randaugment_n, magnitude=randaugment_m)
        )
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return transforms.Compose(transform_list)


def get_eval_transforms(image_size: int = 384) -> transforms.Compose:
    """Deterministic evaluation transforms."""
    return transforms.Compose([
        transforms.Resize(
            int(image_size * 1.14),  # 256 for 224, 438 for 384
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================================
# Dataset
# ============================================================================

class ISICDataset(Dataset):
    """ISIC 2019 Dataset with metadata and optional segmentation masks.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: image_path, label, and metadata columns.
    transform : Optional[transforms.Compose]
        Image transforms.
    use_metadata : bool
        Whether to load and return metadata.
    mask_dir : Optional[Path]
        Directory containing segmentation masks.
    image_size : int
        Target image size (for mask resizing).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        use_metadata: bool = False,
        mask_dir: Optional[Path] = None,
        image_size: int = 384,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_metadata = use_metadata
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Load image
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Label
        label = int(row["label"])
        
        output = {
            "image": image,
            "label": label,
        }
        
        # Metadata
        if self.use_metadata:
            age = row.get("age_norm", 0.0)
            sex = row.get("sex_idx", SEX_VOCAB["unknown"])
            site = row.get("site_idx", SITE_VOCAB["unknown"])
            
            output["metadata"] = {
                "age": torch.tensor(age, dtype=torch.float32),
                "sex": torch.tensor(sex, dtype=torch.long),
                "site": torch.tensor(site, dtype=torch.long),
            }
        
        # Segmentation mask
        if self.mask_dir is not None:
            mask_path = self.mask_dir / f"{row['image']}_segmentation.png"
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize((self.image_size, self.image_size), Image.BILINEAR)
                mask = torch.from_numpy(np.array(mask)).float() / 255.0
                mask = mask.unsqueeze(0)  # (1, H, W)
            else:
                mask = torch.zeros(1, self.image_size, self.image_size)
            output["mask"] = mask
        
        return output


class ISICDatasetTTA(Dataset):
    """ISIC Dataset with TTA - returns multiple augmented versions.
    
    TTA transforms: original, hflip, vflip, crops
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 384,
        use_metadata: bool = False,
        tta_transforms: List[str] = ["original", "hflip", "vflip"],
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.use_metadata = use_metadata
        self.tta_transforms = tta_transforms
        
        # Base transforms
        self.resize = transforms.Resize(
            int(image_size * 1.14),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        self.center_crop = transforms.CenterCrop(image_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        # Corner crops
        self.tl_crop = transforms.Lambda(lambda x: TF.crop(x, 0, 0, image_size, image_size))
        self.tr_crop = transforms.Lambda(
            lambda x: TF.crop(x, 0, x.width - image_size, image_size, image_size)
        )
        self.bl_crop = transforms.Lambda(
            lambda x: TF.crop(x, x.height - image_size, 0, image_size, image_size)
        )
        self.br_crop = transforms.Lambda(
            lambda x: TF.crop(x, x.height - image_size, x.width - image_size, image_size, image_size)
        )

    def __len__(self) -> int:
        return len(self.df)

    def _apply_tta(self, image: Image.Image) -> torch.Tensor:
        """Apply TTA transforms and stack results."""
        images = []
        
        # Resize first
        image = self.resize(image)
        
        for tta in self.tta_transforms:
            if tta == "original":
                img = self.center_crop(image)
            elif tta == "hflip":
                img = TF.hflip(self.center_crop(image))
            elif tta == "vflip":
                img = TF.vflip(self.center_crop(image))
            elif tta == "crop_center":
                img = self.center_crop(image)
            elif tta == "crop_tl":
                img = TF.crop(image, 0, 0, self.image_size, self.image_size)
            elif tta == "crop_br":
                h, w = image.height, image.width
                img = TF.crop(image, h - self.image_size, w - self.image_size, 
                             self.image_size, self.image_size)
            else:
                img = self.center_crop(image)
            
            img = self.normalize(self.to_tensor(img))
            images.append(img)
        
        return torch.stack(images, dim=0)  # (num_tta, C, H, W)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        image = Image.open(row["image_path"]).convert("RGB")
        images = self._apply_tta(image)
        
        label = int(row["label"])
        
        output = {
            "images": images,  # (num_tta, C, H, W)
            "label": label,
        }
        
        if self.use_metadata:
            age = row.get("age_norm", 0.0)
            sex = row.get("sex_idx", SEX_VOCAB["unknown"])
            site = row.get("site_idx", SITE_VOCAB["unknown"])
            
            output["metadata"] = {
                "age": torch.tensor(age, dtype=torch.float32),
                "sex": torch.tensor(sex, dtype=torch.long),
                "site": torch.tensor(site, dtype=torch.long),
            }
        
        return output


# ============================================================================
# Data Loading
# ============================================================================

def _parse_groundtruth_csv(csv_path: Path) -> pd.DataFrame:
    """Parse one-hot ground truth CSV, filtering out UNK."""
    df = pd.read_csv(csv_path)
    
    all_classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    one_hot_cols = [c for c in all_classes if c in df.columns]
    df["dx"] = df[one_hot_cols].idxmax(axis=1)
    
    # Remove UNK
    df = df[df["dx"].isin(VALID_CLASSES)].copy()
    df["label"] = df["dx"].map(LABEL_TO_IDX)
    
    return df[["image", "dx", "label"]]


def _load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load and preprocess metadata CSV."""
    df = pd.read_csv(metadata_path)
    
    # Lesion ID for patient-level split
    if "lesion_id" not in df.columns:
        df["lesion_id"] = df["image"]
    
    # Age normalization (0-1 scale, assuming max age ~90)
    if "age_approx" in df.columns:
        df["age_norm"] = df["age_approx"].fillna(50).clip(0, 90) / 90.0
    else:
        df["age_norm"] = 0.5
    
    # Sex encoding
    if "sex" in df.columns:
        df["sex_idx"] = df["sex"].fillna("unknown").str.lower().map(
            lambda x: SEX_VOCAB.get(x, SEX_VOCAB["unknown"])
        )
    else:
        df["sex_idx"] = SEX_VOCAB["unknown"]
    
    # Anatomical site encoding
    if "anatom_site_general" in df.columns:
        df["site_idx"] = df["anatom_site_general"].fillna("unknown").str.lower().map(
            lambda x: SITE_VOCAB.get(x, SITE_VOCAB["unknown"])
        )
    else:
        df["site_idx"] = SITE_VOCAB["unknown"]
    
    return df


def load_isic_data(
    isic_dir: Path,
    val_ratio: float = 0.2,
    random_state: int = 42,
    fold: Optional[int] = None,
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ISIC data with patient-level split (8-class, no UNK).
    
    Parameters
    ----------
    isic_dir : Path
        ISIC dataset directory.
    val_ratio : float
        Validation ratio (used if fold is None).
    random_state : int
        Random seed.
    fold : Optional[int]
        Fold index for K-fold (0 to n_folds-1). If None, uses simple split.
    n_folds : int
        Number of folds for K-fold.
    
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    isic_dir = Path(isic_dir)
    
    # Paths
    train_gt_path = isic_dir / "ISIC_2019_Training_GroundTruth.csv"
    test_gt_path = isic_dir / "ISIC_2019_Test_GroundTruth.csv"
    train_meta_path = isic_dir / "ISIC_2019_Training_Metadata.csv"
    test_meta_path = isic_dir / "ISIC_2019_Test_Metadata.csv"
    train_img_dir = isic_dir / "ISIC_2019_Training_Input"
    test_img_dir = isic_dir / "ISIC_2019_Test_Input"
    
    # Parse ground truth (removes UNK)
    train_df = _parse_groundtruth_csv(train_gt_path)
    test_df = _parse_groundtruth_csv(test_gt_path)
    
    # Load metadata
    train_meta = _load_metadata(train_meta_path)
    test_meta = _load_metadata(test_meta_path)
    
    # Merge
    train_df = train_df.merge(train_meta, on="image", how="left")
    test_df = test_df.merge(test_meta, on="image", how="left")
    
    # Fill missing
    train_df["lesion_id"] = train_df["lesion_id"].fillna(train_df["image"])
    test_df["lesion_id"] = test_df["lesion_id"].fillna(test_df["image"])
    
    # Image paths
    train_df["image_path"] = train_df["image"].apply(lambda x: train_img_dir / f"{x}.jpg")
    test_df["image_path"] = test_df["image"].apply(lambda x: test_img_dir / f"{x}.jpg")
    
    # Filter existing files
    train_df = train_df[train_df["image_path"].apply(lambda p: p.exists())].copy()
    test_df = test_df[test_df["image_path"].apply(lambda p: p.exists())].copy()
    
    print(f"[Data] Training images (8-class, no UNK): {len(train_df)}")
    print(f"[Data] Test images (8-class, no UNK): {len(test_df)}")
    
    # Patient-level split
    if fold is not None:
        # K-fold
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(sgkf.split(
            train_df["image"].values,
            train_df["label"].values,
            groups=train_df["lesion_id"].values,
        ))
        train_idx, val_idx = splits[fold]
    else:
        # Simple split
        n_splits = int(1.0 / val_ratio)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, val_idx = next(sgkf.split(
            train_df["image"].values,
            train_df["label"].values,
            groups=train_df["lesion_id"].values,
        ))
    
    val_df = train_df.iloc[val_idx].copy().reset_index(drop=True)
    train_df = train_df.iloc[train_idx].copy().reset_index(drop=True)
    
    # Verify no overlap
    train_lesions = set(train_df["lesion_id"].unique())
    val_lesions = set(val_df["lesion_id"].unique())
    overlap = train_lesions & val_lesions
    print(f"[Data] Patient-level split: {len(overlap)} overlapping lesion_ids (should be 0)")
    
    return train_df, val_df, test_df


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """Compute balanced class weights."""
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(labels),
    )
    return torch.tensor(weights, dtype=torch.float32)


def print_class_distribution(df: pd.DataFrame, split_name: str) -> None:
    """Print class distribution."""
    print(f"\n[{split_name}] Class Distribution ({len(df)} samples):")
    for idx, name in enumerate(VALID_CLASSES):
        count = (df["label"] == idx).sum()
        pct = 100.0 * count / len(df) if len(df) > 0 else 0
        print(f"  {idx} {name:5s}: {count:6,} ({pct:5.2f}%)")


# ============================================================================
# DataLoader builders
# ============================================================================

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders.
    
    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        DataFrames with data.
    config : dict
        Configuration dict.
    
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    aug_cfg = config.get("augmentation", {})
    
    image_size = model_cfg.get("image_size", 384)
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = config.get("num_workers", 0)
    pin_memory = config.get("pin_memory", False)
    use_metadata = model_cfg.get("metadata", {}).get("enabled", False)
    
    # Transforms
    train_transform = get_train_transforms(
        image_size=image_size,
        use_randaugment=aug_cfg.get("randaugment", {}).get("enabled", True),
        randaugment_n=aug_cfg.get("randaugment", {}).get("num_ops", 2),
        randaugment_m=aug_cfg.get("randaugment", {}).get("magnitude", 9),
        crop_scale=tuple(aug_cfg.get("random_resized_crop", {}).get("scale", [0.7, 1.0])),
        crop_ratio=tuple(aug_cfg.get("random_resized_crop", {}).get("ratio", [0.9, 1.1])),
        rotation=aug_cfg.get("rotation", 30),
    )
    eval_transform = get_eval_transforms(image_size)
    
    # Datasets
    train_ds = ISICDataset(
        train_df,
        transform=train_transform,
        use_metadata=use_metadata,
        image_size=image_size,
    )
    val_ds = ISICDataset(
        val_df,
        transform=eval_transform,
        use_metadata=use_metadata,
        image_size=image_size,
    )
    test_ds = ISICDataset(
        test_df,
        transform=eval_transform,
        use_metadata=use_metadata,
        image_size=image_size,
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader


def build_tta_dataloader(
    df: pd.DataFrame,
    config: dict,
) -> DataLoader:
    """Build TTA dataloader."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    tta_cfg = config.get("tta", {})
    
    image_size = model_cfg.get("image_size", 384)
    batch_size = train_cfg.get("batch_size", 16) // 2  # Smaller due to multiple augments
    use_metadata = model_cfg.get("metadata", {}).get("enabled", False)
    tta_transforms = tta_cfg.get("transforms", ["original", "hflip", "vflip"])
    
    dataset = ISICDatasetTTA(
        df,
        image_size=image_size,
        use_metadata=use_metadata,
        tta_transforms=tta_transforms,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )


# ============================================================================
# Collate function for metadata
# ============================================================================

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batches with metadata."""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    
    output = {"image": images, "label": labels}
    
    # Metadata
    if "metadata" in batch[0]:
        metadata = {
            "age": torch.stack([item["metadata"]["age"] for item in batch]),
            "sex": torch.stack([item["metadata"]["sex"] for item in batch]),
            "site": torch.stack([item["metadata"]["site"] for item in batch]),
        }
        output["metadata"] = metadata
    
    # Masks
    if "mask" in batch[0]:
        masks = torch.stack([item["mask"] for item in batch])
        output["mask"] = masks
    
    return output


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    isic_dir = Path("./ISIC")
    if isic_dir.exists():
        train_df, val_df, test_df = load_isic_data(isic_dir)
        
        print_class_distribution(train_df, "Train")
        print_class_distribution(val_df, "Val")
        print_class_distribution(test_df, "Test")
        
        # Verify no UNK
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            assert "UNK" not in df["dx"].values, f"UNK found in {name}!"
        print("\n✓ No UNK samples in any split")
        
        # Test dataset
        print("\n=== Testing dataset ===")
        transform = get_train_transforms(image_size=384)
        ds = ISICDataset(train_df.head(10), transform=transform, use_metadata=True)
        
        sample = ds[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        if "metadata" in sample:
            print(f"Metadata: age={sample['metadata']['age']:.2f}, "
                  f"sex={sample['metadata']['sex']}, site={sample['metadata']['site']}")
        
        # Test class weights
        weights = compute_class_weights(train_df["label"].tolist())
        print(f"\nClass weights: {weights}")
