# 🍓 Complete Raspberry Pi Setup Guide for Beginners

**Step-by-step guide to set up Raspberry Pi with Camera Module for AI object detection**

---

## 📋 Table of Contents

1. [What You Need](#what-you-need)
2. [Step 1: Setting Up Raspberry Pi](#step-1-setting-up-raspberry-pi)
3. [Step 2: Connecting the Camera](#step-2-connecting-the-camera)
4. [Step 3: First Boot and WiFi Setup](#step-3-first-boot-and-wifi-setup)
5. [Step 4: Installing Required Software](#step-4-installing-required-software)
6. [Step 5: Setting Up Your Project](#step-5-setting-up-your-project)
7. [Step 6: Running the Detection](#step-6-running-the-detection)
8. [Troubleshooting](#troubleshooting)
9. [Common Questions](#common-questions)

---

## 🛒 What You Need

### Required Hardware:
- **Raspberry Pi 4** (4GB RAM minimum, 8GB recommended)
- **Raspberry Pi Camera Module v2** (or v3)
- **MicroSD Card** (32GB or larger, Class 10 or better)
- **Power Supply** (Official Raspberry Pi 5V 3A USB-C power supply)
- **HDMI Cable** (to connect to monitor/TV)
- **USB Keyboard and Mouse**
- **Monitor or TV** (with HDMI input)
- **Ethernet Cable** (optional, for internet connection)

### Optional but Recommended:
- **MicroSD Card Reader** (to write OS to SD card on your computer)
- **Case for Raspberry Pi** (for protection)
- **Heat Sinks** (for better cooling)

---

## 📦 Step 1: Setting Up Raspberry Pi

### 1.1 Download Raspberry Pi Imager

1. **On your computer**, go to: https://www.raspberrypi.com/software/
2. Download **Raspberry Pi Imager** for your operating system (Windows/Mac/Linux)
3. Install the software

### 1.2 Write Raspberry Pi OS to SD Card

1. **Insert your MicroSD card** into your computer (using card reader)
2. **Open Raspberry Pi Imager**
3. Click **"Choose OS"**
4. Select **"Raspberry Pi OS (recommended)"** - this is the latest version
5. Click **"Choose Storage"** and select your MicroSD card
6. **IMPORTANT**: Click the gear icon (⚙️) to open advanced options:
   - ✅ Enable SSH (so you can connect remotely later)
   - ✅ Set username: `pi` (or your preferred username)
   - ✅ Set password: Choose a strong password
   - ✅ Configure WiFi: Enter your WiFi network name and password
   - ✅ Set locale settings: Choose your country/timezone
7. Click **"Write"** and wait for it to finish (this takes 5-10 minutes)
8. When done, **safely eject** the SD card from your computer

### 1.3 Insert SD Card into Raspberry Pi

1. **Turn off** Raspberry Pi (if it's on)
2. **Insert the MicroSD card** into the slot on the bottom of the Raspberry Pi
3. Make sure it clicks into place

---

## 📷 Step 2: Connecting the Camera

### 2.1 Physical Connection

**⚠️ IMPORTANT: Turn off Raspberry Pi before connecting camera!**

1. **Locate the camera connector** on the Raspberry Pi:
   - It's a small black connector near the HDMI ports
   - It has a small tab that you can lift

2. **Prepare the camera ribbon cable**:
   - The ribbon cable has a connector on one end
   - Make sure the **metal contacts face down** (toward the Raspberry Pi board)
   - The blue/teal side should face up

3. **Connect the camera**:
   - **Lift the black tab** on the camera connector (gently pull up)
   - **Insert the ribbon cable** into the connector
   - Make sure it's **fully inserted** (all the way in)
   - **Push down the black tab** to lock it in place
   - The cable should be **straight and secure**

4. **Secure the cable**:
   - Use the camera case or tape to secure the cable so it doesn't come loose

### 2.2 Verify Connection

- The ribbon cable should be **straight** (not bent at sharp angles)
- The connector should be **fully seated** (no gaps)
- The black tab should be **locked down**

---

## 🚀 Step 3: First Boot and WiFi Setup

### 3.1 First Boot

1. **Connect everything**:
   - HDMI cable to monitor/TV
   - USB keyboard and mouse
   - Power supply (plug into Raspberry Pi)
   - Ethernet cable (optional, if you didn't configure WiFi in Imager)

2. **Power on**:
   - Plug the power supply into the wall
   - Raspberry Pi will start booting (red LED will light up)
   - You'll see the Raspberry Pi logo on screen

3. **Wait for boot** (takes 1-2 minutes on first boot)

### 3.2 Initial Setup (if not done in Imager)

If you didn't configure WiFi in Raspberry Pi Imager:

1. **Click the WiFi icon** (top right corner)
2. **Select your WiFi network**
3. **Enter your WiFi password**
4. Wait for connection (WiFi icon will show signal bars)

### 3.3 Update the System

1. **Open Terminal** (click the terminal icon in the top menu, or press `Ctrl+Alt+T`)
2. **Type these commands** (press Enter after each):

```bash
sudo apt update
```

Wait for it to finish, then:

```bash
sudo apt upgrade -y
```

⚠️ **This takes 10-30 minutes** - be patient! It's updating all software.

---

## 🔧 Step 4: Installing Required Software

### 4.1 Enable Camera Interface

**This is VERY IMPORTANT!** The camera won't work without this step.

1. **Open Terminal** (if not already open)
2. **Type this command**:

```bash
sudo raspi-config
```

3. **Navigate using arrow keys**:
   - Use **↓** to go down to **"3 Interface Options"**
   - Press **Enter**
   - Use **↓** to go down to **"I1 Legacy Camera"** or **"I3 Camera"**
   - Press **Enter**
   - Select **"Yes"** to enable camera
   - Press **Enter**
   - Press **Tab** to select **"Finish"**
   - Press **Enter**

4. **Reboot Raspberry Pi**:
   ```bash
   sudo reboot
   ```

5. **Wait for reboot** (1-2 minutes)

### 4.2 Install Python and Camera Libraries

1. **Open Terminal** (after reboot)
2. **Install system packages**:

```bash
sudo apt install -y python3-picamera2 python3-pip python3-opencv
```

Wait for installation (2-5 minutes)

### 4.3 Install Python Packages

1. **Navigate to your project folder**:

```bash
cd ~/smart_road/my_model
```

2. **Install Python packages**:

```bash
pip3 install -r requirements.txt
```

Or install individually:
```bash
pip3 install ultralytics opencv-python numpy pillow pyrebase4
```

Wait for installation (5-10 minutes - this downloads AI libraries)

---

## 📁 Step 5: Setting Up Your Project

### 5.1 Create Project Folder

1. **Open File Manager** (folder icon in top menu)
2. **Navigate to your home folder** (usually `/home/pi`)
3. **Create a new folder** called `smart_road`:
   - Right-click → New Folder
   - Name it: `smart_road`

### 5.2 Copy Your Files

You need to copy these files to your Raspberry Pi:

**Option A: Using USB Drive**
1. Copy files from your computer to a USB drive
2. Insert USB drive into Raspberry Pi
3. Copy files from USB to `smart_road` folder

**Option B: Using File Transfer (SCP/SFTP)**
1. Use FileZilla or WinSCP on your computer
2. Connect to Raspberry Pi using:
   - Host: `raspberrypi.local` or your Pi's IP address
   - Username: `pi` (or your username)
   - Password: your password
3. Copy files to `smart_road` folder

**Option C: Using Git (if you have internet)**
```bash
cd ~
git clone <your-repository-url> smart_road
```

### 5.3 Required Files

Make sure these files are in your `smart_road/my_model/` folder:
- ✅ `raspberry_pi_yolo_client.py` - Main detection script
- ✅ `my_model.pt` - Your trained YOLO model
- ✅ `requirements.txt` - Python dependencies

### 5.4 Verify Files

1. **Open Terminal**
2. **Navigate to project folder**:

```bash
cd ~/smart_road/my_model
```

3. **List files**:

```bash
ls -la
```

You should see:
- `raspberry_pi_yolo_client.py`
- `my_model.pt`
- `requirements.txt`

---

## 🎯 Step 6: Running the Detection

### 6.1 Test Camera First

Before running detection, test if camera works:

```bash
libcamera-hello --list-cameras
```

You should see your camera listed.

### 6.2 Configure Detection Script

1. **Open the detection script**:

```bash
nano raspberry_pi_yolo_client.py
```

2. **Check these settings** (around line 25-30):
   ```python
   MODEL_PATH = "my_model.pt"  # Make sure this matches your model file name
   CONFIDENCE_THRESHOLD = 0.5  # Detection threshold (0.0 to 1.0)
   CAMERA_WIDTH = 640
   CAMERA_HEIGHT = 480
   SHOW_DISPLAY = True  # Set False if no monitor connected
   ```

3. **Save and exit**:
   - Press `Ctrl+X`
   - Press `Y` to confirm
   - Press `Enter`

### 6.3 Run Detection

1. **Make sure you're in the right folder**:

```bash
cd ~/smart_road/my_model
```

2. **Run the detection script**:

```bash
python3 raspberry_pi_yolo_client.py
```

3. **What to expect**:
   - Camera will initialize (takes 2-3 seconds)
   - Model will load (takes 5-10 seconds first time)
   - Detection window will open showing camera feed
   - Objects will be detected and shown with bounding boxes
   - Press `Q` to quit

### 6.4 Running Without Monitor (Headless)

If you don't have a monitor connected:

1. **Edit the script**:

```bash
nano raspberry_pi_yolo_client.py
```

2. **Change this line**:
   ```python
   SHOW_DISPLAY = False  # Changed from True
   ```

3. **Save and run** (detection will still work, just no window)

---

## 🔍 Troubleshooting

### Camera Not Working?

**Problem**: Camera initialization fails

**Solutions**:
1. **Check camera is enabled**:
   ```bash
   sudo raspi-config
   # Interface Options > Camera > Enable
   sudo reboot
   ```

2. **Check camera connection**:
   - Make sure ribbon cable is fully inserted
   - Check cable isn't damaged
   - Try reconnecting the cable

3. **Test camera**:
   ```bash
   libcamera-hello -t 0
   ```
   This should show camera preview for 5 seconds

### Model Not Found?

**Problem**: `Model file not found: my_model.pt`

**Solutions**:
1. **Check file location**:
   ```bash
   ls -la my_model.pt
   ```
   Should show the file

2. **Check file name** matches in script:
   ```python
   MODEL_PATH = "my_model.pt"  # Must match actual file name
   ```

3. **Use full path** if needed:
   ```python
   MODEL_PATH = "/home/pi/smart_road/my_model/my_model.pt"
   ```

### Slow Performance?

**Problem**: Detection is very slow

**Solutions**:
1. **Reduce resolution**:
   ```python
   CAMERA_WIDTH = 320
   CAMERA_HEIGHT = 240
   ```

2. **Increase capture interval**:
   ```python
   CAPTURE_INTERVAL = 0.5  # Slower = 2 FPS
   ```

3. **Close other applications**

4. **Use Raspberry Pi 4** (Pi 3 is slower)

### Import Errors?

**Problem**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
1. **Reinstall packages**:
   ```bash
   pip3 install --upgrade ultralytics opencv-python numpy pillow
   ```

2. **Use pip3, not pip**:
   ```bash
   pip3 install <package-name>
   ```

### Permission Denied?

**Problem**: Permission errors when running script

**Solutions**:
1. **Don't use sudo** (usually not needed):
   ```bash
   python3 raspberry_pi_yolo_client.py  # Correct
   sudo python3 raspberry_pi_yolo_client.py  # Usually wrong
   ```

2. **Check file permissions**:
   ```bash
   chmod +x raspberry_pi_yolo_client.py
   ```

---

## ❓ Common Questions

### Q: Do I need internet connection?

**A**: 
- **For initial setup**: Yes, to download software
- **For running detection**: No, once everything is installed, it works offline

### Q: Can I use Raspberry Pi 3?

**A**: Yes, but it will be slower. Raspberry Pi 4 (4GB+) is recommended for best performance.

### Q: How do I connect remotely (SSH)?

**A**: 
1. Make sure SSH is enabled (done in Raspberry Pi Imager)
2. Find your Pi's IP address:
   ```bash
   hostname -I
   ```
3. On your computer, use:
   - **Windows**: PuTTY or Windows Terminal
   - **Mac/Linux**: Terminal
   - Connect to: `pi@raspberrypi.local` or `pi@<IP-ADDRESS>`

### Q: Can I use a different camera?

**A**: This guide is for Raspberry Pi Camera Module. USB cameras require different code.

### Q: How do I stop the detection?

**A**: Press `Q` in the detection window, or press `Ctrl+C` in terminal

### Q: Can I run detection automatically on boot?

**A**: Yes, you can set up a systemd service. This is advanced - ask for help if needed.

### Q: How much storage do I need?

**A**: 
- **OS**: ~8GB
- **Software**: ~5GB
- **Model file**: ~50-200MB
- **Total**: 32GB SD card is minimum, 64GB+ recommended

### Q: My Pi is getting hot, is that normal?

**A**: 
- **Warm (40-50°C)**: Normal
- **Hot (60-70°C)**: Consider adding heat sinks or fan
- **Very hot (80°C+)**: Add cooling immediately

---

## ✅ Quick Reference Commands

```bash
# Enable camera
sudo raspi-config
# Interface Options > Camera > Enable

# Update system
sudo apt update && sudo apt upgrade -y

# Install camera libraries
sudo apt install -y python3-picamera2 python3-pip python3-opencv

# Install Python packages
pip3 install ultralytics opencv-python numpy pillow

# Test camera
libcamera-hello -t 0

# Run detection
cd ~/smart_road/my_model
python3 raspberry_pi_yolo_client.py

# Check Pi temperature
vcgencmd measure_temp

# Check IP address
hostname -I
```

---

## 🎉 You're Done!

Your Raspberry Pi is now set up and ready to run AI object detection!

**Next Steps**:
1. Test the detection script
2. Adjust settings for your needs
3. Integrate with your ESP32 Master controller
4. Set up automatic startup (optional)

**Need Help?**
- Check the troubleshooting section above
- Review error messages carefully
- Make sure all steps were completed

---

**Good luck with your Smart Road project!** 🚗🛣️

