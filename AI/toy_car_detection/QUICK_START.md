# 🚀 Quick Start Guide

Follow these steps in order while you collect images and label data.

## Step 0: Install Python (If Needed)

**If you get "python is not recognized":**

1. Run: `toy_car_detection\INSTALL_PYTHON.bat` (opens download page)
2. Or see: `INSTALL_PYTHON.md` for detailed instructions
3. Download Python 3.11 from https://www.python.org/downloads/windows/
4. **IMPORTANT**: Check "Add Python to PATH" during installation!
5. Restart your terminal after installation

## Step 1: Setup Environment (Do This First!)

```bash
py toy_car_detection/setup_environment.bat
```

This will:
- ✅ Upgrade pip
- ✅ Install PyTorch with CUDA 12.1 (for RTX 4060)
- ✅ Install YOLO, Flask, and other dependencies
- ✅ Test installation

**Time: ~5-10 minutes**

---

## Step 2: Collect Images (Do This in Parallel!)

While setup runs, start collecting images:

1. **Use ESP32-CAM** to take photos of toy cars:
   - Different angles
   - Different lighting
   - Different distances
   - Different backgrounds

2. **Download images** from Google (free/royalty-free):
   - Search: "toy police car", "toy ambulance", etc.
   - Aim for 100-300 images per class

3. **Save images** temporarily in a folder (we'll organize later)

**Target: 100-300 images per class (police_car, ambulance, normal_car)**

---

## Step 3: Label Images with Roboflow

1. Go to [roboflow.com](https://roboflow.com) and sign up (free)

2. **Create Project**:
   - Click "Create New Project"
   - Name: "Toy Car Detection"
   - Type: **Object Detection**
   - Click "Create Project"

3. **Upload Images**:
   - Click "Upload" or drag & drop
   - Upload all your images

4. **Label Images**:
   - Click on an image
   - Draw bounding boxes around each toy car
   - Assign class: `police_car`, `ambulance`, or `normal_car`
   - Save and move to next image

5. **Split Dataset**:
   - Roboflow will auto-split train/valid
   - Or manually split: 80% train, 20% valid

6. **Export**:
   - Click "Export"
   - Format: **YOLOv8**
   - Click "Continue" → Download ZIP

7. **Extract Dataset**:
   ```bash
   # Extract the ZIP to toy_car_detection/dataset/
   # Should have structure:
   # dataset/
   #   ├── train/
   #   │   ├── images/
   #   │   └── labels/
   #   ├── valid/
   #   │   ├── images/
   #   │   └── labels/
   #   └── data.yaml
   ```

---

## Step 4: Train Model

Once dataset is ready:

```bash
cd toy_car_detection
py train.py
```

**Time: ~30-60 minutes** (depending on dataset size and GPU)

After training:
- ✅ Best model: `runs/detect/train/weights/best.pt`
- ✅ Check results in `runs/detect/train/`

---

## Step 5: Start Detection Server

```bash
cd toy_car_detection
py server.py
```

Or use the batch file:
```bash
py toy_car_detection/start_server.bat
```

Server will run on `http://0.0.0.0:5000`

**Test it:**
```bash
# In another terminal
py toy_car_detection/test_server.py path/to/test_image.jpg
```

---

## Step 6: Configure ESP32-CAM

1. **Find your laptop's IP**:
   ```bash
   ipconfig
   # Look for IPv4 Address (e.g., 192.168.1.100)
   ```

2. **Update `esp32_cam.ino`**:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   String serverUrl = "http://192.168.1.100:5000/detect";  // Your IP!
   ```

3. **Upload to ESP32-CAM**:
   - Open `toy_car_detection/esp32_cam.ino` in Arduino IDE
   - Select board: **AI Thinker ESP32-CAM**
   - Upload sketch

4. **Monitor Serial** (115200 baud):
   - You should see detections every second!

---

## 🎯 Complete Workflow

```
1. Setup Environment (5-10 min)
   ↓
2. Collect Images (ongoing)
   ↓
3. Label with Roboflow (1-2 hours)
   ↓
4. Train Model (30-60 min)
   ↓
5. Start Server
   ↓
6. Configure ESP32-CAM
   ↓
7. 🎉 Done! Detections streaming!
```

---

## 💡 Tips

- **Start with fewer images** (50 per class) to test the pipeline
- **Add more images** later to improve accuracy
- **Use YOLOv8n** (nano) for speed, **YOLOv8s** (small) for accuracy
- **Monitor training** - stop early if overfitting
- **Test server** before connecting ESP32-CAM

---

## 🆘 Troubleshooting

**"Model not found" error:**
- Train first: `py train.py`
- Check `MODEL_PATH` in `server.py`

**ESP32 can't connect:**
- Check WiFi credentials
- Verify server IP
- Ensure server is running
- Check Windows Firewall (allow port 5000)

**Poor accuracy:**
- More training images
- More epochs
- Larger image size
- Better quality labels

---

## 📞 Next Steps

Once working:
1. Tune confidence threshold
2. Adjust camera frame rate
3. Add more classes
4. Create web dashboard
5. Deploy to production

Good luck! 🚀


