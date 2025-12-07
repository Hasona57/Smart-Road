# 🚗 Toy Car Detection with ESP32-CAM

YOLOv8-based toy car detection system using ESP32-CAM and RTX 4060 GPU.

## 📋 Quick Start

### 1. Setup Python Environment

Run the setup script:
```bash
py toy_car_detection/setup_environment.bat
```

Or manually:
```bash
py -m pip install --upgrade pip
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
py -m pip install -r requirements.txt
```

### 2. Collect & Label Dataset

1. **Take photos** of toy cars:
   - 🚓 Police cars
   - 🚑 Ambulances  
   - 🚗 Normal cars

2. **Label with Roboflow**:
   - Go to [roboflow.com](https://roboflow.com)
   - Create new project → Object Detection
   - Upload images
   - Draw bounding boxes around each car
   - Export as **YOLOv8 format**

3. **Organize dataset**:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```

4. **Create `data.yaml`** (copy from `data.yaml.template`):
   ```yaml
   path: dataset
   train: train/images
   val: valid/images
   
   names:
     0: police_car
     1: ambulance
     2: normal_car
   ```

### 3. Train Model

```bash
py toy_car_detection/train.py
```

Training will:
- Use YOLOv8n (nano - fastest)
- Train for 50 epochs
- Save best model to `runs/detect/train/weights/best.pt`

### 4. Start Detection Server

```bash
py toy_car_detection/server.py
```

Server will listen on `http://0.0.0.0:5000`

### 5. Configure & Upload ESP32-CAM Code

1. **Find your laptop's IP**:
   ```bash
   ipconfig
   ```
   Look for IPv4 address (e.g., `192.168.1.100`)

2. **Update `esp32_cam.ino`**:
   - Set `ssid` and `password`
   - Set `serverUrl` to `http://YOUR_IP:5000/detect`

3. **Upload to ESP32-CAM**:
   - Open in Arduino IDE
   - Select board: **AI Thinker ESP32-CAM**
   - Upload sketch

4. **Monitor Serial** (115200 baud) to see detections!

## 📁 Project Structure

```
toy_car_detection/
├── server.py              # Flask server for ESP32-CAM
├── train.py               # Training script
├── esp32_cam.ino          # ESP32-CAM Arduino code
├── dataset/
│   ├── train/             # Training images & labels
│   ├── valid/             # Validation images & labels
│   └── data.yaml          # Dataset config
└── README.md              # This file
```

## 🎯 API Endpoints

### `POST /detect`
Receives JPEG image, returns JSON detections.

**Request:**
- `multipart/form-data`
- Field: `image` (JPEG bytes)

**Response:**
```json
{
  "detections": [
    {
      "label": "police_car",
      "confidence": 0.92,
      "x1": 10.5,
      "y1": 20.3,
      "x2": 150.7,
      "y2": 180.2
    }
  ],
  "count": 1
}
```

### `GET /health`
Health check endpoint.

## ⚙️ Configuration

### Training Parameters
Edit `train.py` to adjust:
- `epochs`: Training epochs (default: 50)
- `imgsz`: Image size (default: 320)
- `batch`: Batch size (default: 16)

### Server Parameters
Edit `server.py`:
- `MODEL_PATH`: Path to trained model
- `conf`: Confidence threshold (default: 0.25)
- `imgsz`: Inference size (default: 320)

### ESP32-CAM Parameters
Edit `esp32_cam.ino`:
- `captureInterval`: Time between captures (ms)
- `FRAMESIZE_QVGA`: Camera resolution (QVGA = 320x240)

## 🔧 Troubleshooting

### Model not found
- Train first: `py toy_car_detection/train.py`
- Check `MODEL_PATH` in `server.py`

### ESP32 can't connect
- Check WiFi credentials
- Verify server IP address
- Ensure server is running
- Check firewall allows port 5000

### Poor detection accuracy
- Collect more training images (100-300 per class)
- Increase training epochs
- Use larger image size (640 instead of 320)
- Try YOLOv8s instead of YOLOv8n

## 📊 Performance Tips

- **Speed**: Use YOLOv8n, imgsz=320, QVGA camera
- **Accuracy**: Use YOLOv8s, imgsz=640, VGA camera
- **GPU**: Ensure CUDA is working: `python -c "import torch; print(torch.cuda.is_available())"`

## 🎓 Next Steps

1. Add more classes (fire truck, etc.)
2. Implement real-time video streaming
3. Add web dashboard for visualization
4. Deploy to edge device (Jetson Nano, etc.)

## 📝 License

MIT License - Feel free to use and modify!


