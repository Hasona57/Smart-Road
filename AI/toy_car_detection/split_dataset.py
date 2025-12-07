"""
Split dataset into train/validation sets.
Moves a percentage of images from train to valid.
"""
import os
import shutil
from pathlib import Path
import random

def split_dataset(dataset_dir, train_ratio=0.8, seed=42):
    """
    Split dataset into train/validation sets.
    
    Args:
        dataset_dir: Path to dataset directory
        train_ratio: Ratio of data to keep in training (default 0.8 = 80%)
        seed: Random seed for reproducibility
    """
    dataset_path = Path(dataset_dir)
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    valid_images = dataset_path / "valid" / "images"
    valid_labels = dataset_path / "valid" / "labels"
    
    # Create valid directories if they don't exist
    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(train_images.glob("*.jpg")) + list(train_images.glob("*.png"))
    image_files = [f for f in image_files if f.is_file()]
    
    if len(image_files) == 0:
        print("❌ No images found in train/images!")
        return False
    
    print(f"📊 Found {len(image_files)} images in training set")
    
    # Check if valid already has images
    existing_valid = list(valid_images.glob("*.jpg")) + list(valid_images.glob("*.png"))
    if len(existing_valid) > 0:
        print(f"⚠️  Warning: {len(existing_valid)} images already in valid/images")
        response = input("Do you want to continue? This will move more images. (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return False
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    valid_files = image_files[split_idx:]
    
    print(f"📦 Splitting dataset:")
    print(f"   Training: {len(train_files)} images ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"   Validation: {len(valid_files)} images ({len(valid_files)/len(image_files)*100:.1f}%)")
    
    if len(valid_files) == 0:
        print("❌ Not enough images to create validation set!")
        print("   Need at least 2 images to split.")
        return False
    
    # Move files to validation set
    moved = 0
    for img_file in valid_files:
        # Move image
        label_file = train_labels / (img_file.stem + ".txt")
        valid_img = valid_images / img_file.name
        valid_lbl = valid_labels / (img_file.stem + ".txt")
        
        if img_file.exists():
            shutil.move(str(img_file), str(valid_img))
            moved += 1
        
        # Move corresponding label if it exists
        if label_file.exists():
            shutil.move(str(label_file), str(valid_lbl))
    
    print(f"✅ Moved {moved} images and labels to validation set")
    print(f"📁 Training: {len(list(train_images.glob('*.jpg')) + list(train_images.glob('*.png')))} images")
    print(f"📁 Validation: {len(list(valid_images.glob('*.jpg')) + list(valid_images.glob('*.png')))} images")
    
    return True

if __name__ == "__main__":
    import sys
    
    # Default to toy_car_detection/dataset
    script_dir = Path(__file__).parent
    default_dataset = script_dir / "dataset"
    
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
    else:
        dataset_path = default_dataset
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        print(f"💡 Usage: py split_dataset.py [dataset_path]")
        sys.exit(1)
    
    print("=" * 50)
    print("Dataset Splitter")
    print("=" * 50)
    print(f"Dataset: {dataset_path}")
    print()
    
    success = split_dataset(dataset_path, train_ratio=0.8)
    
    if success:
        print()
        print("=" * 50)
        print("✅ Dataset split complete!")
        print("=" * 50)
        print("💡 You can now run training: py toy_car_detection/train.py")
    else:
        print()
        print("❌ Dataset split failed!")
        sys.exit(1)

