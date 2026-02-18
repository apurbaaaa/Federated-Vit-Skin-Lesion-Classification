"""
Data utilities for ISIC 2019 skin lesion classification (8-class, no UNK).

Features:
    - Patient-level split (StratifiedGroupKFold)
    - UNK class removed before splitting
    - Metadata loading and encoding
    - Precomputed segmentation mask loading (4-channel input)
    - SOTA augmentation (RandAugment, ColorJitter, etc.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
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

class JointTransform:
    """Apply transforms to both image and mask consistently."""
    
    def __init__(
        self,
        image_size: int = 224,
        is_train: bool = True,
        use_randaugment: bool = True,
        randaugment_n: int = 2,
        randaugment_m: int = 9,
        color_jitter: bool = True,
        rotation: int = 30,
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.use_randaugment = use_randaugment and HAS_RANDAUGMENT
        self.color_jitter = color_jitter
        self.rotation = rotation
        
        if color_jitter and is_train:
            self.color_jitter_transform = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            )
        else:
            self.color_jitter_transform = None
        
        if self.use_randaugment and is_train:
            self.randaugment = RandAugment(
                num_ops=randaugment_n,
                magnitude=randaugment_m,
            )
        else:
            self.randaugment = None
    
    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.is_train:
            return self._train_transform(image, mask)
        else:
            return self._eval_transform(image, mask)
    
    def _train_transform(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Random crop parameters
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image,
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
        )
        
        # Apply crop
        image = TF.resized_crop(
            image, i, j, h, w, 
            (self.image_size, self.image_size),
            interpolation=TF.InterpolationMode.BICUBIC,
        )
        if mask is not None:
            mask = TF.resized_crop(
                mask, i, j, h, w,
                (self.image_size, self.image_size),
                interpolation=TF.InterpolationMode.NEAREST,
            )
        
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
        
        # Random vertical flip
        if torch.rand(1) < 0.5:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)
        
        # Random rotation
        angle = float(torch.empty(1).uniform_(-self.rotation, self.rotation).item())
        image = TF.rotate(image, angle)
        if mask is not None:
            mask = TF.rotate(mask, angle)
        
        # Color jitter (image only)
        if self.color_jitter_transform is not None:
            image = self.color_jitter_transform(image)
        
        # RandAugment (image only)
        if self.randaugment is not None:
            image = self.randaugment(image)
        
        # To tensor
        image_tensor = TF.to_tensor(image)  # (3, H, W)
        image_tensor = TF.normalize(
            image_tensor,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        
        mask_tensor = None
        if mask is not None:
            mask_tensor = TF.to_tensor(mask)  # (1, H, W)
            mask_tensor = (mask_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return image_tensor, mask_tensor
    
    def _eval_transform(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Resize
        resize_size = int(self.image_size * 1.14)
        image = TF.resize(
            image, resize_size,
            interpolation=TF.InterpolationMode.BICUBIC,
        )
        if mask is not None:
            mask = TF.resize(
                mask, resize_size,
                interpolation=TF.InterpolationMode.NEAREST,
            )
        
        # Center crop
        image = TF.center_crop(image, self.image_size)
        if mask is not None:
            mask = TF.center_crop(mask, self.image_size)
        
        # To tensor
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        
        mask_tensor = None
        if mask is not None:
            mask_tensor = TF.to_tensor(mask)
            mask_tensor = (mask_tensor - 0.5) / 0.5
        
        return image_tensor, mask_tensor


# ============================================================================
# Dataset
# ============================================================================

class ISICDataset(Dataset):
    """ISIC 2019 Dataset with optional precomputed segmentation masks.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: image_path, label, and metadata columns.
    image_size : int
        Target image size.
    is_train : bool
        Whether this is training set.
    use_metadata : bool
        Whether to load and return metadata.
    use_segmentation_mask : bool
        Whether to load precomputed masks and concatenate as 4th channel.
    mask_dir : Optional[Path]
        Directory containing precomputed masks.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        is_train: bool = True,
        use_metadata: bool = False,
        use_segmentation_mask: bool = False,
        mask_dir: Optional[Path] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.is_train = is_train
        self.use_metadata = use_metadata
        self.use_segmentation_mask = use_segmentation_mask
        self.mask_dir = Path(mask_dir) if mask_dir else None
        
        self.transform = JointTransform(
            image_size=image_size,
            is_train=is_train,
            use_randaugment=is_train,
        )
        
        if use_segmentation_mask and mask_dir is None:
            raise ValueError("mask_dir must be provided when use_segmentation_mask=True")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Load image
        image = Image.open(row["image_path"]).convert("RGB")
        
        # Load mask if enabled
        mask = None
        if self.use_segmentation_mask:
            image_id = row["image"]
            mask_path = self.mask_dir / f"{image_id}.png"
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for {image_id}. "
                    f"Expected at: {mask_path}. "
                    f"Run precompute_masks.py first."
                )
            mask = Image.open(mask_path).convert("L")
        
        # Apply joint transform
        image_tensor, mask_tensor = self.transform(image, mask)
        
        # Concatenate mask as 4th channel
        if self.use_segmentation_mask and mask_tensor is not None:
            image_tensor = torch.cat([image_tensor, mask_tensor], dim=0)  # (4, H, W)
        
        label = int(row["label"])
        
        output = {
            "image": image_tensor,
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
    
    if "lesion_id" not in df.columns:
        df["lesion_id"] = df["image"]
    
    if "age_approx" in df.columns:
        df["age_norm"] = df["age_approx"].fillna(50).clip(0, 90) / 90.0
    else:
        df["age_norm"] = 0.5
    
    if "sex" in df.columns:
        df["sex_idx"] = df["sex"].fillna("unknown").str.lower().map(
            lambda x: SEX_VOCAB.get(x, SEX_VOCAB["unknown"])
        )
    else:
        df["sex_idx"] = SEX_VOCAB["unknown"]
    
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ISIC data with patient-level split (8-class, no UNK).
    
    Parameters
    ----------
    isic_dir : Path
        ISIC dataset directory.
    val_ratio : float
        Validation ratio.
    random_state : int
        Random seed.
    
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
    
    print(f"[Data] Training images (8-class, no UNK): {len(train_df)}")
    print(f"[Data] Test images (8-class, no UNK): {len(test_df)}")
    
    # Load metadata
    train_meta = _load_metadata(train_meta_path)
    test_meta = _load_metadata(test_meta_path)
    
    # Merge
    train_df = train_df.merge(train_meta, on="image", how="left")
    test_df = test_df.merge(test_meta, on="image", how="left")
    
    # Add image paths
    train_df["image_path"] = train_df["image"].apply(
        lambda x: str(train_img_dir / f"{x}.jpg")
    )
    test_df["image_path"] = test_df["image"].apply(
        lambda x: str(test_img_dir / f"{x}.jpg")
    )
    
    # Patient-level split using lesion_id
    groups = train_df["lesion_id"].fillna(train_df["image"])
    
    # Simple stratified split
    kfold = StratifiedGroupKFold(
        n_splits=int(1 / val_ratio) if val_ratio > 0 else 5,
        shuffle=True,
        random_state=random_state,
    )
    train_idx, val_idx = next(kfold.split(train_df, train_df["label"], groups))
    
    val_df = train_df.iloc[val_idx].copy()
    train_df = train_df.iloc[train_idx].copy()
    
    # Verify no overlap
    train_lesions = set(train_df["lesion_id"].dropna())
    val_lesions = set(val_df["lesion_id"].dropna())
    overlap = train_lesions & val_lesions
    print(f"[Data] Patient-level split: {len(overlap)} overlapping lesion_ids (should be 0)")
    
    return train_df, val_df, test_df


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """Compute balanced class weights."""
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    full_weights = np.ones(NUM_CLASSES)
    for c, w in zip(classes, weights):
        full_weights[c] = w
    return torch.tensor(full_weights, dtype=torch.float32)


def print_class_distribution(df: pd.DataFrame, split_name: str) -> None:
    """Print class distribution."""
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    print(f"\n[{split_name}] Class Distribution ({total} samples):")
    for idx in range(NUM_CLASSES):
        count = counts.get(idx, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"  {idx} {VALID_CLASSES[idx]:5s}: {count:6,} ({pct:5.2f}%)")


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders."""
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    
    image_size = model_cfg.get("image_size", 224)
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)
    use_metadata = model_cfg.get("metadata", {}).get("enabled", False)
    use_segmentation_mask = data_cfg.get("use_segmentation_mask", False)
    mask_dir = data_cfg.get("segmentation_mask_dir", "./masks")
    
    train_dataset = ISICDataset(
        df=train_df,
        image_size=image_size,
        is_train=True,
        use_metadata=use_metadata,
        use_segmentation_mask=use_segmentation_mask,
        mask_dir=mask_dir,
    )
    
    val_dataset = ISICDataset(
        df=val_df,
        image_size=image_size,
        is_train=False,
        use_metadata=use_metadata,
        use_segmentation_mask=use_segmentation_mask,
        mask_dir=mask_dir,
    )
    
    test_dataset = ISICDataset(
        df=test_df,
        image_size=image_size,
        is_train=False,
        use_metadata=use_metadata,
        use_segmentation_mask=use_segmentation_mask,
        mask_dir=mask_dir,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return train_loader, val_loader, test_loader
