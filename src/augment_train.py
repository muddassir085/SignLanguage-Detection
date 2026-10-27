"""Augment the training dataset by applying realistic image transformations."""

import argparse
import sys
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd

from landmarks import create_holistic, extract_features_from_rgb, feature_names

def augment_image(image: np.ndarray, seed: int) -> np.ndarray:
    np.random.seed(seed)
    random.seed(seed)
    
    h, w = image.shape[:2]
    
    # 1. Rotation (-10 to 10 degrees)
    angle = random.uniform(-10, 10)
    
    # 2. Scale / Zoom (0.9 to 1.1)
    scale = random.uniform(0.9, 1.1)
    
    # 3. Translation (+- 5-10%)
    tx = random.uniform(-0.1, 0.1) * w
    ty = random.uniform(-0.1, 0.1) * h
    
    # Build affine transform matrix
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    aug_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # 4. Brightness (+- 15%) & 5. Contrast (+- 10-15%)
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-38, 38)
    aug_image = cv2.convertScaleAbs(aug_image, alpha=alpha, beta=beta)
    
    # 6. Slight Gaussian Blur (optional, ~30% chance)
    if random.random() < 0.3:
        aug_image = cv2.GaussianBlur(aug_image, (3, 3), 0)
        
    # 7. Small Image Noise (optional, ~30% chance)
    if random.random() < 0.3:
        noise = np.random.normal(0, 3, aug_image.shape).astype(np.float32)
        aug_image = cv2.add(aug_image.astype(np.float32), noise)
        aug_image = np.clip(aug_image, 0, 255).astype(np.uint8)
        
    return aug_image

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-train", type=Path, default=Path("dataset/processed/original/train.csv"))
    parser.add_argument("--output-train", type=Path, default=Path("dataset/processed/train.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load original training data
    df = pd.read_csv(args.original_train)
    
    # Prepare list for new rows
    augmented_rows = []
    
    # Class multiplier logic
    # Larger classes ~2x (1 orig + 1 aug)
    # Smaller classes ~3x (1 orig + 2 aug)
    class_multipliers = {
        "hello": 1,
        "ok": 1,
        "thank_you": 1,
        "i_love_you": 2,
        "peace": 2,
        "yes": 2
    }

    print(f"Original training data size: {len(df)}")
    
    columns = feature_names()
    unreadable = 0
    failed_extraction = 0
    
    with create_holistic(static_image_mode=True) as holistic:
        for idx, row in df.iterrows():
            label = row["label"]
            source_path = row["source_path"]
            source_group = row["source_group"]
            
            # Read original image
            image_path = args.raw_dir / source_path
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"Warning: Could not read {image_path}", file=sys.stderr)
                unreadable += 1
                continue
                
            num_augmentations = class_multipliers.get(label, 1)
            
            for aug_idx in range(num_augmentations):
                # Generate unique seed per augmentation to ensure reproducibility but variability
                aug_seed = args.seed + idx * 10 + aug_idx
                
                aug_image = augment_image(image, seed=aug_seed)
                rgb = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
                
                extraction = extract_features_from_rgb(rgb, holistic)
                
                # Create a new row
                new_row = dict(zip(columns, extraction.features.tolist(), strict=True))
                new_row.update(
                    label=label,
                    source_path=f"{source_path}_aug{aug_idx}",
                    source_group=source_group # VERY IMPORTANT: Keep same source group
                )
                augmented_rows.append(new_row)
                
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} / {len(df)} original images...")
                
    if unreadable > 0:
        print(f"Warning: {unreadable} images were unreadable.")
        
    aug_df = pd.DataFrame(augmented_rows)
    
    # Combine original and augmented
    combined_df = pd.concat([df, aug_df], ignore_index=True)
    
    # Save to output
    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(args.output_train, index=False)
    
    print(f"Augmented training data size: {len(combined_df)}")
    print("Class distribution after augmentation:")
    print(combined_df["label"].value_counts())
    
if __name__ == "__main__":
    main()
