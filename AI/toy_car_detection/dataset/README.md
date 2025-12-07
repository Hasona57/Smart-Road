# Dataset Directory

Place your training and validation data here.

## Structure

```
dataset/
├── train/
│   ├── images/     # Training images (.jpg, .png)
│   └── labels/     # Training labels (.txt in YOLO format)
├── valid/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
└── data.yaml       # Dataset configuration (create from data.yaml.template)
```

## YOLO Label Format

Each image should have a corresponding `.txt` file with the same name.

Format: `class_id center_x center_y width height` (normalized 0-1)

Example (`image001.txt`):
```
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.3
```

Where:
- `0` = police_car
- `1` = ambulance  
- `2` = normal_car

## Getting Started

1. Export your labeled dataset from Roboflow as YOLOv8 format
2. Extract the ZIP contents here
3. Copy `data.yaml` from `data.yaml.template` and update if needed
4. Run training: `py toy_car_detection/train.py`


