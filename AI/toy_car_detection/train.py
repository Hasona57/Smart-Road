"""
Train YOLOv8 model for toy car detection.
Run this after preparing your dataset with Roboflow or manual labeling.
"""
import os
import yaml
import tempfile
from pathlib import Path
from ultralytics import YOLO

def main():
    print("🚗 Toy Car Detection - YOLOv8 Training")
    print("=" * 50)
    
    # Check if data.yaml exists (try multiple possible locations)
    script_dir = Path(__file__).parent
    data_yaml = script_dir / "dataset" / "data.yaml"
    if not data_yaml.exists():
        # Try relative to current working directory
        data_yaml = Path("dataset/data.yaml")
        if not data_yaml.exists():
            # Try absolute path from script location
            data_yaml = Path.cwd() / "toy_car_detection" / "dataset" / "data.yaml"
    
    if not data_yaml.exists():
        print(f"❌ Error: {data_yaml} not found!")
        print("\n💡 Create data.yaml with this structure:")
        print("""
path: dataset
train: train/images
val: valid/images

names:
  0: police_car
  1: ambulance
  2: normal_car
        """)
        return
    
    print(f"✅ Found dataset config: {data_yaml}")
    
    # Check if validation images exist
    data_yaml_abs = data_yaml.resolve()
    dataset_dir = data_yaml_abs.parent
    valid_images_dir = dataset_dir / "valid" / "images"
    
    valid_images = list(valid_images_dir.glob("*.jpg")) + list(valid_images_dir.glob("*.png"))
    if len(valid_images) == 0:
        print("\n⚠️  WARNING: No validation images found!")
        print(f"   Validation directory: {valid_images_dir}")
        print("\n💡 You need to split your dataset into train/validation sets.")
        print("   Run this command to automatically split:")
        print(f"   py toy_car_detection/split_dataset.py")
        print("\n   Or manually move some images from train/images to valid/images")
        return
    
    # Read and update data.yaml
    with open(data_yaml_abs, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Update path to absolute path
    data_config['path'] = str(dataset_dir)
    
    # Write updated config to a temporary file
    temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(data_config, temp_yaml)
    temp_yaml.close()
    data_yaml_path = temp_yaml.name
    
    # Load base model (YOLOv8n = nano, smallest and fastest)
    model_name = "yolov8n.pt"
    print(f"\n📦 Loading base model: {model_name}")
    model = YOLO(model_name)
    
    # Training parameters optimized for RTX 4060
    print("\n🎯 Training Configuration:")
    print("  Model: YOLOv8n (nano - fastest)")
    print("  Image size: 320x320 (good for small toy cars)")
    print("  Epochs: 50")
    print("  Batch size: 16")
    print("  Device: GPU (CUDA) if available")
    print()
    
    # Start training
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=50,
            imgsz=320,
            batch=16,
            device=0,  # Use GPU (0) if available, else CPU
            project="runs/detect",
            name="train",
            exist_ok=True,
            patience=10,  # Early stopping if no improvement
            save=True,
            plots=True,
        )
        
        print("\n" + "=" * 50)
        print("✅ Training completed!")
        print("=" * 50)
        print(f"\n📁 Best model saved to: runs/detect/train/weights/best.pt")
        print(f"📁 Last model saved to: runs/detect/train/weights/last.pt")
        print("\n💡 Next steps:")
        print("  1. Check training results in runs/detect/train/")
        print("  2. Update server.py MODEL_PATH if needed")
        print("  3. Start server: py toy_car_detection/server.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


