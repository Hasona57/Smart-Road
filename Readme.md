# Smart Road Eye: AI-Powered Traffic Management System

## Project Overview

**Smart Road Eye** is an intelligent traffic management system that uses AI vision processing to monitor road conditions in real-time, analyze traffic patterns, and automatically control traffic signals. The system integrates Raspberry Pi for AI-powered object detection, ESP32 for traffic control, and Firebase Realtime Database for cloud connectivity, providing a complete smart traffic solution.

### Key Features

- **AI-Powered Vision System**: Raspberry Pi with Camera Module runs YOLO object detection locally to analyze traffic density and vehicle types
- **Intelligent Traffic Control**: ESP32 Master controller adjusts traffic light timing based on AI analysis and sensor data
- **Real-time Cloud Integration**: Firebase Realtime Database stores and synchronizes data across all system components
- **Multi-Sensor Integration**: Air pollution monitoring, speed detection, vehicle detection, and day/night sensing
- **Automated Actuators**: Servo-controlled emergency gates, speed bumps, and pedestrian gates
- **Mobile App Integration**: Flutter mobile app displays real-time road status and pollution levels for drivers
- **Smart Street Lighting**: Automatic street light control based on ambient light conditions
- **Direct I2C LCD Display**: LCD connected directly to ESP32 (no Arduino needed)

---

## System Architecture

### Hardware Components

#### 1. AI Vision System (Raspberry Pi)
- **Raspberry Pi 4** (8GB+ RAM) - Main AI processing unit
- **Raspberry Pi Camera Module v2** - High-quality road monitoring camera
- **MicroSD Card** (64GB+) - Operating system and software storage
- Runs YOLO object detection model locally for real-time traffic analysis

#### 2. Traffic Control System (ESP32 Master)
- **ESP32 Development Board** - Main traffic controller
- **Traffic Light Module** - Physical 3-color LED traffic signal
- **3x Servo Motors** - Emergency gate, speed bump, and pedestrian gate control
- **5V Relay Module** - Street light control (10 white LEDs)
- **I2C LCD Display 16x2** - System status display (directly connected to ESP32)

#### 3. Sensor Network
- **MQ135 Gas Sensor** - Air pollution monitoring
- **2x IR Sensors** - Vehicle speed detection (entry/exit points)
- **HC-SR04 Ultrasonic Sensor** - Vehicle presence detection
- **LDR Module** - Day/night detection for automatic lighting
- **Push Button** - Pedestrian crossing request

#### 4. Communication
- **WiFi Connectivity** - Both Raspberry Pi and ESP32 connect to WiFi

### Software Stack

- **Raspberry Pi**: Python 3 with YOLO (Ultralytics), OpenCV, Picamera2, Pyrebase4
- **ESP32 Master**: Arduino C++ with Firebase ESP32, ESP32Servo, LiquidCrystal_I2C libraries
- **Cloud**: Firebase Realtime Database
- **Mobile App**: Flutter (Dart) with Firebase integration

---

## How It Works

### 1. AI Vision Processing
```
Raspberry Pi Camera → YOLO Detection → Object Classification → Firebase Upload
```
- Camera captures road images continuously
- YOLO model detects and classifies vehicles (cars, trucks, buses, etc.)
- Detection results sent to Firebase every second
- Traffic density calculated from detection counts

### 2. Traffic Control Logic
```
Firebase Data → ESP32 Master → Sensor Reading → Traffic Light Control
```
- ESP32 Master reads AI detection data from Firebase
- Combines with local sensor data (pollution, speed, vehicle presence)
- Calculates optimal traffic light timing for each lane
- Controls physical traffic lights and actuators
- Displays status on I2C LCD directly connected to ESP32

### 3. Data Flow
```
Raspberry Pi → Firebase → ESP32 Master → Traffic Lights
                ↓                          ↓
            Mobile App (Flutter)      I2C LCD Display
```

### 4. Sensor Integration
- **Air Pollution**: MQ135 sensor monitors air quality, data sent to Firebase
- **Speed Detection**: Two IR sensors measure vehicle speed, trigger speed bump for violations
- **Vehicle Detection**: Ultrasonic sensor detects vehicle presence at intersection
- **Day/Night**: LDR sensor automatically controls street lighting

---

## Project Structure

```
Smart-Road/
├── code/
│   ├── ESP32_Master/
│   │   └── esp32_master.ino          # Main traffic controller code
│   ├── my_model/
│   │   ├── raspberry_pi_yolo_client.py  # AI detection client (Raspberry Pi)
│   │   ├── my_model.pt                # Trained YOLO model
│   │   ├── requirements.txt            # Python dependencies
│   │   └── README.md                  # Raspberry Pi detection guide
│   ├── Application Flutter/           # You can access the final APK through Smart-Road\Code\Application flutter\build\app\outputs\flutter-apk
│   │   ├── lib/
│   │   │   ├── main.dart              # Flutter app main file
│   │   │   ├── models/
│   │   │   │   └── road_data.dart     # Data models
│   │   │   └── services/
│   │   │       ├── background_monitor.dart      # Background monitoring
│   │   │       ├── notification_service.dart    # Push notifications
│   │   │       └── road_cache_manager.dart      # Data caching
│   │   ├── assets/
│   │   │   └── sounds/                # App sound files
│   │   ├── pubspec.yaml               # Flutter dependencies
│   │   └── README.md                  # Flutter app documentation
│   ├── Firebase/
│   │   ├── firebase_config.h.example  # Firebase configuration template
│   │   └── firebase_rules.json        # Database security rules
│   ├── Labels/
│   │   ├── Hassan/                    # Training images (Hassan folder)
│   │   └── [Training images]          # Additional training data
│   ├── WIRING_CONNECTIONS.md          # Complete wiring guide
│   └── RASPBERRY_PI_SETUP_GUIDE.md   # Beginner setup guide
├── AI/
│   ├── toy_car_detection/             # YOLO model training project
│   │   ├── train.py                   # Training script
│   │   ├── dataset/                   # Training dataset
│   │   └── [YOLO model files]          # Pre-trained models
│   ├── src/
│   │   ├── detector.py                # Detection script
│   │   └── api_detector.py            # API detection script
│   └── requirements.txt               # AI training dependencies
├── 3D/
│   ├── Printer/                       # 3D printable STL files
│   └── [3D model files]                # Component 3D models (STEP files)
├── Images/                             # Project images and screenshots
├── BOM.csv                             # Bill of Materials
├── README.md                           # This file (main documentation)
└── JOURNAL.md                          # Project development journal
```

---

## Installation & Setup

### Prerequisites

- Raspberry Pi 4 (8GB+ RAM recommended)
- ESP32 Development Board
- All sensors and components (see BOM.csv)
- WiFi network access
- Firebase account and project

### Step 1: Raspberry Pi Setup

1. **Install Raspberry Pi OS**
   - Download Raspberry Pi Imager
   - Flash OS to MicroSD card
   - Enable SSH and configure WiFi during setup

2. **Enable Camera**
   ```bash
   sudo raspi-config
   # Interface Options > Camera > Enable
   sudo reboot
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3-picamera2 python3-pip python3-opencv
   ```

4. **Install Python Packages**
   ```bash
   cd code/my_model
   pip3 install -r requirements.txt
   ```

5. **Configure and Run Detection**
   ```bash
   # Edit raspberry_pi_yolo_client.py to set model path
   python3 raspberry_pi_yolo_client.py
   ```

**Detailed Setup**: See [code/RASPBERRY_PI_SETUP_GUIDE.md](code/RASPBERRY_PI_SETUP_GUIDE.md) for complete beginner instructions.

### Step 2: ESP32 Master Setup

1. **Install Arduino IDE**
   - Download from https://www.arduino.cc/en/software
   - Install ESP32 board support:
     - File > Preferences > Additional Board Manager URLs
     - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
     - Tools > Board > Boards Manager > Search "ESP32" > Install

2. **Install Required Libraries**
   - Tools > Manage Libraries
   - Install:
     - Firebase ESP32 Client
     - ESP32Servo
     - LiquidCrystal_I2C (by Frank de Brabander)

3. **Configure WiFi and Firebase**
   - Edit `code/ESP32_Master/esp32_master.ino`
   - Update WiFi credentials:
     ```cpp
     #define WIFI_SSID "YOUR_WIFI_SSID"
     #define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
     ```
   - Update Firebase credentials:
     ```cpp
     #define FIREBASE_HOST "YOUR_FIREBASE_URL"
     #define FIREBASE_AUTH "YOUR_FIREBASE_AUTH"
     ```
   - Update I2C LCD address if needed (default: 0x27):
     ```cpp
     #define LCD_I2C_ADDRESS 0x27  // Change to 0x3F if your LCD uses different address
     ```

4. **Upload Code**
   - Connect ESP32 via USB
   - Select board: Tools > Board > ESP32 Dev Module
   - Select port: Tools > Port > [Your ESP32 Port]
   - Click Upload

### Step 3: Firebase Setup

1. **Create Firebase Project**
   - Go to https://console.firebase.google.com/
   - Create new project
   - Enable Realtime Database

2. **Get Credentials**
   - Project Settings > General > Your apps
   - Copy Database URL and Web API Key

3. **Set Database Rules**
   - Database > Rules
   - Use rules from `code/Firebase/firebase_rules.json`

### Step 4: Hardware Assembly

1. **Follow Wiring Guide**
   - See [code/WIRING_CONNECTIONS.md](code/WIRING_CONNECTIONS.md) for complete pin-to-pin connections

2. **I2C LCD Connection**
   - Connect LCD SDA to ESP32 GPIO 21
   - Connect LCD SCL to ESP32 GPIO 22
   - Connect LCD VCC to ESP32 5V (or 3.3V depending on module)
   - Connect LCD GND to ESP32 GND
   - **Note**: I2C LCD is connected directly to ESP32 (no Arduino needed)

3. **Power Supply**
   - Raspberry Pi: 5V 3A USB-C power supply
   - ESP32: USB or external 5V supply
   - Servos: External 5V 3A supply (recommended)

4. **Camera Connection**
   - Connect Raspberry Pi Camera Module ribbon cable
   - Ensure proper orientation (metal contacts down)

---

## Configuration

### Raspberry Pi Detection Settings

Edit `code/my_model/raspberry_pi_yolo_client.py`:

```python
MODEL_PATH = "my_model.pt"           # Your YOLO model file
CONFIDENCE_THRESHOLD = 0.5           # Detection threshold (0.0-1.0)
CAMERA_WIDTH = 640                   # Image width
CAMERA_HEIGHT = 480                  # Image height
CAPTURE_INTERVAL = 0.3              # Time between captures (seconds)
SEND_TO_FIREBASE = True              # Enable Firebase updates
FIREBASE_UPDATE_INTERVAL = 1.0      # Update interval (seconds)
```

### ESP32 Master Settings

Edit `code/ESP32_Master/esp32_master.ino`:

```cpp
#define WIFI_SSID "YOUR_WIFI"
#define WIFI_PASSWORD "YOUR_PASSWORD"
#define FIREBASE_HOST "YOUR_FIREBASE_URL"
#define FIREBASE_AUTH "YOUR_FIREBASE_AUTH"
#define LCD_I2C_ADDRESS 0x27
```

---

## Firebase Database Structure

```
smart-traffic-system/
├── ai/
│   └── detections/
│       ├── latest/          # Latest detection results
│       ├── history/          # Detection history
│       └── summary/         # Summary statistics
├── road_status/
│   ├── traffic/
│   │   ├── lane1/          # Lane 1 state (red/yellow/green)
│   │   ├── lane2/          # Lane 2 state
│   │   └── lane3/          # Lane 3 state
│   ├── sensors/
│   │   ├── pollution_ppm/   # Air pollution level
│   │   ├── daytime/        # Day/night status
│   │   └── vehicle_detected/ # Vehicle presence
│   └── system/
│       ├── emergency_mode/  # Emergency mode flag
│       ├── speed_violations/ # Speed violation count
│       └── failsafe_mode/   # Failsafe status
└── system/
    ├── esp32_status/        # ESP32 online status
    └── raspberry_pi_status/ # Raspberry Pi online status
```

---

## Usage

### Starting the System

1. **Power On Raspberry Pi**
   ```bash
   cd ~/Smart-Road/code/my_model
   python3 raspberry_pi_yolo_client.py
   ```

2. **Power On ESP32 Master**
   - Connect via USB or external power
   - ESP32 will automatically:
     - Connect to WiFi
     - Connect to Firebase
     - Initialize I2C LCD display
     - Start reading sensor data
     - Control traffic lights

3. **Monitor System**
   - Check Firebase console for real-time data
   - View I2C LCD display for system status
   - Use Flutter mobile app for driver view

### System Operation

- **Normal Mode**: AI analyzes traffic, ESP32 adjusts light timing
- **Emergency Mode**: All lanes turn green for emergency direction
- **Speed Violation**: Speed bump activates automatically
- **Pedestrian Request**: Button press triggers pedestrian crossing sequence
- **Night Mode**: Street lights turn on automatically
- **LCD Display**: Shows current lane status, remaining time, and system messages

---

## Mobile App

The Flutter mobile application provides:
- Real-time traffic density visualization
- Air pollution level monitoring
- Traffic light status for all lanes
- Route recommendations based on congestion
- Emergency alerts and notifications

**Location**: `code/Application flutter/`

---

## Troubleshooting

### Raspberry Pi Issues

**Camera not working:**
```bash
sudo raspi-config  # Enable camera
sudo reboot
libcamera-hello -t 0  # Test camera
```

**Model not found:**
- Verify `my_model.pt` exists in `code/my_model/` folder
- Check file permissions

**Firebase connection failed:**
- Check WiFi connection
- Verify Firebase credentials
- Check database rules allow writes

### ESP32 Issues

**WiFi connection failed:**
- Verify SSID and password
- Check 2.4GHz network (ESP32 doesn't support 5GHz)
- Check signal strength

**Firebase connection failed:**
- Verify Firebase URL and auth key
- Check database rules
- Ensure WiFi is connected first

**I2C LCD not displaying:**
- Check I2C connections (SDA → GPIO 21, SCL → GPIO 22)
- Verify I2C address (try 0x27 or 0x3F)
- Use I2C scanner to find correct address
- Check power supply (5V or 3.3V depending on module)
- Ensure common ground connection

**Sensors not reading:**
- Check wiring connections
- Verify pin assignments in code
- Test sensors individually

---

## Bill of Materials

See [BOM.csv](BOM.csv) for complete component list with prices and vendors.

**Approximate Total Cost**: ~8835 EGP (Egyptian Pounds)

**Note**: Arduino has been removed from the system. I2C LCD is now connected directly to ESP32.

---

## Safety Considerations

- **Power Supply**: Use appropriate current ratings for all components
- **Voltage Levels**: ESP32 uses 3.3V logic, ensure proper level shifting if needed
- **Servo Motors**: Use external power supply for multiple servos
- **Battery Safety**: Follow proper battery handling procedures
- **Firebase Security**: Use proper database rules to prevent unauthorized access
- **I2C Pull-ups**: Most I2C LCD modules have built-in pull-up resistors

---

## Project Development Journey

### Phase 1: Ideation
- Goal: Reduce accidents and traffic congestion
- Concept: AI vision monitoring + smart traffic control

### Phase 2: Hardware Design
- Custom PCB design using KiCad
- Component selection and footprint research
- Power system design with LDO regulators

### Phase 3: Software Development
- YOLO model training and validation
- Firebase integration
- ESP32 control logic implementation

### Phase 4: Integration
- Raspberry Pi migration from ESP32-CAM
- System integration and testing
- Mobile app development
- **I2C LCD direct connection** (removed Arduino dependency)

### Phase 5: Refinement
- Performance optimization
- Error handling and failsafe modes
- Documentation and user guides

---

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## License

This project is open source and available under the MIT License.

---

## Authors

**Hasona57**
- GitHub: [@Hasona57](https://github.com/Hasona57)

---

## Acknowledgments

- ESP32 community for hardware support and libraries
- Raspberry Pi Foundation for excellent hardware and documentation
- Firebase team for real-time database services
- Ultralytics for YOLO object detection framework
- OpenCV community for computer vision tools
- Flutter team for mobile app framework

---

## Future Enhancements

- [ ] Machine learning model optimization for better accuracy
- [ ] Additional sensor integration (weather, noise levels)
- [ ] Multi-intersection coordination
- [ ] Advanced traffic flow prediction
- [ ] Integration with traffic management APIs
- [ ] Web dashboard for traffic management
- [ ] Edge computing optimization

---

## Support

For detailed setup instructions:
- **Raspberry Pi**: See [code/RASPBERRY_PI_SETUP_GUIDE.md](code/RASPBERRY_PI_SETUP_GUIDE.md)
- **Wiring**: See [code/WIRING_CONNECTIONS.md](code/WIRING_CONNECTIONS.md)
- **Raspberry Pi Detection**: See [code/my_model/README.md](code/my_model/README.md)

---

**Smart Road Eye** - Making roads safer and smarter through AI and IoT technology.
**Note** - This README.md is written with a lot of AI help.

