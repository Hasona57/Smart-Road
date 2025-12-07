# Raspberry Pi YOLO Detection System

Real-time object detection using Raspberry Pi Camera Module and YOLO model.

**Runs AI locally on Raspberry Pi - no external server needed!**

---

## 🚀 Quick Start

### 1. Prerequisites

- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi Camera Module v2 or v3
- MicroSD card (32GB+)
- See [RASPBERRY_PI_SETUP_GUIDE.md](../RASPBERRY_PI_SETUP_GUIDE.md) for complete beginner setup

### 2. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install camera libraries
sudo apt install -y python3-picamera2 python3-pip python3-opencv

# Install Python packages
pip3 install -r requirements.txt
# Or install individually:
# pip3 install ultralytics opencv-python numpy pillow pyrebase4
```

### 3. Enable Camera

```bash
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
# Reboot: sudo reboot
```

### 4. Run Detection

```bash
cd my_model
python3 raspberry_pi_yolo_client.py
```

Press `Q` to quit.

---

## ⚙️ Configuration

Edit `raspberry_pi_yolo_client.py` to adjust settings:

```python
MODEL_PATH = "my_model.pt"           # Your model file
CONFIDENCE_THRESHOLD = 0.5           # Detection threshold (0.0-1.0)
CAMERA_WIDTH = 640                   # Image width
CAMERA_HEIGHT = 480                   # Image height
CAPTURE_INTERVAL = 0.3               # Time between captures (seconds)
SHOW_DISPLAY = True                   # Show detection window

# Firebase settings (same database as ESP32 Master)
SEND_TO_FIREBASE = True              # Enable/disable Firebase updates
FIREBASE_UPDATE_INTERVAL = 1.0      # Update Firebase every 1 second
```

---

## 📊 Performance Settings

### For Best Accuracy:
```python
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAPTURE_INTERVAL = 0.5
CONFIDENCE_THRESHOLD = 0.3
```

### For Best Speed:
```python
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAPTURE_INTERVAL = 0.2
CONFIDENCE_THRESHOLD = 0.6
```

### Balanced (Recommended):
```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_INTERVAL = 0.3
CONFIDENCE_THRESHOLD = 0.5
```

---

## 🔧 Troubleshooting

### Camera Not Working
```bash
# Enable camera
sudo raspi-config
# Interface Options > Camera > Enable
sudo reboot

# Test camera
libcamera-hello -t 0
```

### Model Not Found
- Check `my_model.pt` exists in same folder
- Verify `MODEL_PATH` in script matches file name

### Slow Performance
- Reduce resolution (320x240)
- Increase `CAPTURE_INTERVAL` (0.5 or higher)
- Close other applications

### Import Errors
```bash
pip3 install --upgrade ultralytics opencv-python numpy pillow pyrebase4
```

### Firebase Connection Issues
- Check WiFi connection: `ping 8.8.8.8`
- Verify Firebase credentials in code match ESP32 Master
- Check Firebase database rules allow writes
- Set `SEND_TO_FIREBASE = False` to disable if needed

---

## 📁 Files

- `raspberry_pi_yolo_client.py` - Main detection script
- `my_model.pt` - Your trained YOLO model
- `requirements.txt` - Python dependencies

---

## 📚 Complete Setup Guide

For detailed step-by-step instructions for beginners, see:
**[RASPBERRY_PI_SETUP_GUIDE.md](../RASPBERRY_PI_SETUP_GUIDE.md)**

---

## 🔥 Firebase Integration

Detection results are automatically sent to Firebase Realtime Database (same as ESP32 Master):

**Firebase Paths:**
- `/ai/detections/latest` - Latest detection results
- `/ai/detections/history` - Detection history (last 100 entries)
- `/ai/detections/summary` - Summary statistics
- `/system/raspberry_pi_status` - Raspberry Pi online status (1 = online, 0 = offline)

**Detection Data Structure:**
```json
{
  "timestamp": 1234567890,
  "frame_count": 100,
  "inference_time_ms": 250.5,
  "detection_count": 3,
  "detections": [
    {
      "class": "car",
      "class_id": 2,
      "confidence": 0.85,
      "bbox": {"xmin": 100, "ymin": 150, "xmax": 300, "ymax": 400}
    }
  ],
  "class_counts": {"car": 2, "truck": 1}
}
```

**To disable Firebase:**
```python
SEND_TO_FIREBASE = False
```

---

## 🎯 Features

- ✅ Runs YOLO locally on Raspberry Pi
- ✅ Real-time object detection
- ✅ **Sends detections to Firebase Realtime Database**
- ✅ **Same database as ESP32 Master**
- ✅ No external server needed
- ✅ Low latency
- ✅ Supports Raspberry Pi Camera Module v2/v3
- ✅ Configurable detection settings
- ✅ Visual display with bounding boxes

---

**Need help?** Check the troubleshooting section or the complete setup guide!
