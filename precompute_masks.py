#!/usr/bin/env python3
"""
Precompute segmentation masks for ISIC-2019 dataset using classical CV.

Pipeline per image:
  1. Resize to target_size x target_size
  2. Convert to LAB colour space → use L (lightness) channel
  3. GaussianBlur to suppress hair / noise
  4. Otsu threshold (inverted so darker lesion = foreground)
  5. Apply circular ROI mask to suppress dark border / vignette
  6. Morphological close → open to clean up
  7. Keep largest connected component
  8. Light dilation to include lesion border
  9. Save as single-channel PNG (0 = background, 255 = lesion)

Usage:
    python precompute_masks.py --isic_dir ./ISIC --output_dir ./masks
    python precompute_masks.py --isic_dir ./ISIC --output_dir ./masks --workers 4
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core segmentation function (pure OpenCV, no torch dependency)
# ---------------------------------------------------------------------------

def segment_lesion(image_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Classical-CV lesion segmentation for dermoscopy images.

    Args:
        image_bgr: BGR image (any resolution).
        target_size: output mask resolution.

    Returns:
        Binary mask uint8 (0 or 255), shape (target_size, target_size).
    """
    # --- Resize --------------------------------------------------------
    img = cv2.resize(image_bgr, (target_size, target_size),
                     interpolation=cv2.INTER_AREA)

    # --- Colour conversion --------------------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)  # lightness [0, 255]

    # --- Blur to suppress hair / fine texture --------------------------
    blurred = cv2.GaussianBlur(l_ch, (0, 0), sigmaX=5)
    blurred = blurred.astype(np.uint8)

    # --- Otsu threshold (inverted: dark object → white foreground) -----
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- Circular ROI to suppress dark corners / vignetting ------------
    h, w = binary.shape
    cx, cy = w // 2, h // 2
    radius = int(min(h, w) * 0.45)
    circle_mask = np.zeros_like(binary)
    cv2.circle(circle_mask, (cx, cy), radius, 255, thickness=-1)
    binary = cv2.bitwise_and(binary, circle_mask)

    # --- Morphological cleanup -----------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Keep largest connected component ------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels > 1:
        # label 0 is background
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        binary = np.where(labels == largest, 255, 0).astype(np.uint8)
    else:
        # Fallback: no foreground detected → return centered ellipse
        binary = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(binary, (cx, cy), (w // 3, h // 3), 0, 0, 360, 255, -1)

    # --- Slight dilation to include lesion border ----------------------
    binary = cv2.dilate(binary, kernel, iterations=1)

    return binary


# ---------------------------------------------------------------------------
# File-level worker (used by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_single(args: Tuple[Path, Path, int]) -> str:
    """Process a single image file → save mask. Returns image_id."""
    image_path, output_dir, target_size = args
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return f"SKIP:{image_path.stem}"
    mask = segment_lesion(img, target_size=target_size)
    out_path = output_dir / f"{image_path.stem}.png"
    cv2.imwrite(str(out_path), mask)
    return image_path.stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_image_paths(image_dir: Path) -> List[Path]:
    """Collect all image paths from a directory."""
    paths: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(image_dir.glob(ext))
    return sorted(paths)


def precompute_masks(
    image_dir: Path,
    output_dir: Path,
    target_size: int = 224,
    workers: int = 0,
) -> int:
    """Generate masks for all images in image_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_image_paths(image_dir)
    if not paths:
        print(f"  No images found in {image_dir}")
        return 0

    print(f"  Found {len(paths)} images")

    tasks = [(p, output_dir, target_size) for p in paths]
    count = 0
    skipped = 0

    if workers <= 1:
        # Single-process (simpler, useful for debugging)
        for t in tqdm(tasks, desc="  Segmenting"):
            result = _process_single(t)
            if result.startswith("SKIP:"):
                skipped += 1
            else:
                count += 1
    else:
        # Multi-process
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_single, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures),
                           desc="  Segmenting"):
                result = fut.result()
                if result.startswith("SKIP:"):
                    skipped += 1
                else:
                    count += 1

    if skipped:
        print(f"  Skipped {skipped} unreadable images")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Precompute lesion masks (classical CV)"
    )
    parser.add_argument("--isic_dir", type=str, default="./ISIC",
                        help="ISIC dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./masks",
                        help="Output directory for mask PNGs")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Mask output resolution")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = single-process)")
    args = parser.parse_args()

    isic_dir = Path(args.isic_dir)
    output_dir = Path(args.output_dir)

    for split_name, subdir in [
        ("Training", "ISIC_2019_Training_Input"),
        ("Test", "ISIC_2019_Test_Input"),
    ]:
        image_dir = isic_dir / subdir
        if not image_dir.exists():
            print(f"[{split_name}] Skipping – {image_dir} not found")
            continue

        print(f"\n[{split_name}] Processing images from {image_dir}")
        n = precompute_masks(
            image_dir, output_dir,
            target_size=args.image_size,
            workers=args.workers,
        )
        print(f"[{split_name}] Saved {n} masks → {output_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
