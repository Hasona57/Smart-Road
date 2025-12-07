# ESP32-CAM YOLO Detection System

Real-time object detection using ESP32-CAM and YOLO model via Flask server.

**Supports latest YOLO versions: YOLOv8, YOLOv9, YOLOv10, YOLOv11**

## Quick Start

### 1. Install Dependencies

**Windows:**
```cmd
py -m pip install -r requirements.txt
```

**Linux/Mac:**
```bash
pip install -r requirements.txt
```

**Update to latest YOLO:**
```cmd
py -m pip install --upgrade ultralytics
```

### 2. Find Your Computer's IP Address

**Windows:**
```cmd
ipconfig
```
Look for "IPv4 Address" (e.g., `192.168.8.238`)

**Linux/Mac:**
```bash
ifconfig | grep "inet "
```

### 3. Configure ESP32-CAM

1. Open `esp32_yolo_client.ino` in Arduino IDE
2. Update WiFi credentials (lines 21-22):
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. Update server URL (line 26) with your computer's IP:
   ```cpp
   String serverUrl = "http://192.168.8.238:5000/detect";
   ```
4. Select board: **AI Thinker ESP32-CAM**
5. Upload to ESP32-CAM

### 4. Run Flask Server

**Basic:**
```cmd
py flask_yolo_server.py --model my_model.pt
```

**With OpenCV window:**
```cmd
py flask_yolo_server.py --model my_model.pt --display
```

**Optimized for speed:**
```cmd
py flask_yolo_server.py --model my_model.pt --imgsz 480
```

**With GPU (if available):**
```cmd
py flask_yolo_server.py --model my_model.pt --device cuda
```

### 5. View Detections

- **Web Browser**: Open `http://localhost:5000` - Live detection view with bounding boxes
- **OpenCV Window**: Use `--display` flag to show window on your computer
- **Direct Image**: `http://localhost:5000/view`
- **MJPEG Stream**: `http://localhost:5000/stream`

## Server Options

```
--model      Path to YOLO model file (required)
--port       Server port (default: 5000)
--host       Server host (default: 0.0.0.0)
--thresh     Confidence threshold (default: 0.5)
--display    Show OpenCV window with detections
--imgsz      Inference size (default: 640 for accuracy, 480 for speed)
--device     Device: cpu or cuda (default: cpu)
```

## ESP32 Configuration

### Recommended Settings (Good Balance):
```cpp
config.frame_size = FRAMESIZE_QVGA;  // 320x240
config.jpeg_quality = 10;
const int captureInterval = 300;  // 3 FPS
```

### For Better Accuracy:
```cpp
config.frame_size = FRAMESIZE_VGA;  // 640x480
config.jpeg_quality = 8;
const int captureInterval = 500;  // 2 FPS
```

### For Maximum Speed:
```cpp
config.frame_size = FRAMESIZE_QVGA;  // 320x240
config.jpeg_quality = 12;
const int captureInterval = 200;  // 5 FPS
```

## Performance

| Configuration | Resolution | FPS | Latency | Accuracy |
|--------------|------------|-----|---------|----------|
| Balanced | QVGA (320x240) | 3 | 300-500ms | High |
| High Quality | VGA (640x480) | 2 | 400-600ms | Very High |
| Fast | QVGA (320x240) | 5 | 200-400ms | Good |
| GPU Accelerated | VGA (640x480) | 3-5 | 100-300ms | Very High |

## Troubleshooting

### Connection Errors (Error -11)
- **Check server is running**: Verify Flask server is active
- **Verify IP address**: Update `serverUrl` in ESP32 code
- **Same WiFi network**: ESP32 and computer must be on same network
- **Firewall**: Allow Python through Windows Firewall
- **Test connection**: Open `http://YOUR_IP:5000/health` in browser

### Slow Performance
- Use GPU: `--device cuda` (if available)
- Reduce inference size: `--imgsz 480`
- Increase frame interval: `captureInterval = 500`
- Close other applications

### Low Accuracy
- Increase resolution: `FRAMESIZE_VGA`
- Lower JPEG quality: `config.jpeg_quality = 8`
- Increase inference size: `--imgsz 640`
- Lower threshold: `--thresh 0.3`

### Windows Issues
- Use `py` instead of `python`: `py -m pip install -r requirements.txt`
- If `py` doesn't work, try `python`
- Check Python is installed and in PATH

## Files

- `flask_yolo_server.py` - Flask server for YOLO detection (receives images from ESP32-CAM)
- `esp32_yolo_client.ino` - ESP32-CAM client code (sends images to server)
- `my_model.pt` - Your trained YOLO model
- `requirements.txt` - Python dependencies (latest YOLO included)

## API Endpoints

- `POST /detect` - Receive image from ESP32-CAM for detection (returns JSON)
- `GET /view` - View latest detection image
- `GET /stream` - MJPEG stream of detections
- `GET /health` - Health check
- `GET /` - Web interface

## Example Response

```json
{
  "success": true,
  "detections": [
    {
      "class": "car",
      "class_id": 2,
      "confidence": 0.85,
      "bbox": {
        "xmin": 100,
        "ymin": 150,
        "xmax": 300,
        "ymax": 400
      }
    }
  ],
  "count": 1,
  "image_size": {
    "width": 320,
    "height": 240
  }
}
```
