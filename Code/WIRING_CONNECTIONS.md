# 🔌 Smart Road Management System - Detailed Wiring Connections

Complete pin-to-pin wiring guide for all components in the Smart Road Management System.

---

## 📋 Table of Contents

1. [ESP32 Master Controller Connections](#esp32-master-controller-connections)
2. [Raspberry Pi Camera Connections](#raspberry-pi-camera-connections)
3. [I2C LCD Display Connections](#i2c-lcd-display-direct-connection-to-esp32)
4. [Power Distribution](#power-distribution)
5. [Component Interconnections](#component-interconnections)
6. [Connection Diagrams](#connection-diagrams)

---

## 🔧 ESP32 Master Controller Connections

### Traffic Light Module (Single Physical Light - 3 Colors)

**Note:** Only ONE physical traffic light module is used. Other lanes are simulated in code/Firebase/app.

The traffic light module has built-in resistors and 4 pins: Red, Yellow, Green, and GND (common ground).

| Module Pin | ESP32 Pin | Connection | Notes |
|------------|-----------|------------|-------|
| Red | GPIO 2 | Direct connection | Active HIGH, module has built-in resistor |
| Yellow | GPIO 4 | Direct connection | Active HIGH, module has built-in resistor |
| Green | GPIO 5 | Direct connection | Active HIGH, module has built-in resistor |
| GND | ESP32 GND | Direct connection | Common ground |

**Wiring:**
```
Traffic Light Module Red    → ESP32 GPIO 2
Traffic Light Module Yellow → ESP32 GPIO 4
Traffic Light Module Green  → ESP32 GPIO 5
Traffic Light Module GND    → ESP32 GND
```

**Note:** Lanes 2 and 3 traffic light states are simulated in software and displayed in Firebase/mobile app only.

---

### Servo Motors

#### Servo 1 - Emergency Gate
| Pin | Connection | Notes |
|-----|------------|-------|
| Signal (Orange/Yellow) | ESP32 GPIO 12 | PWM control |
| Power (Red) | External 5V (recommended) or ESP32 5V | Use external supply for stability |
| Ground (Black/Brown) | Common GND | Connect to ESP32 GND |

**Wiring:**
```
Servo Signal → ESP32 GPIO 12
Servo VCC    → External 5V Power Supply (or ESP32 5V pin)
Servo GND    → ESP32 GND
```

#### Servo 2 - Speed Bump
| Pin | Connection | Notes |
|-----|------------|-------|
| Signal (Orange/Yellow) | ESP32 GPIO 13 | PWM control |
| Power (Red) | External 5V (recommended) or ESP32 5V | Use external supply for stability |
| Ground (Black/Brown) | Common GND | Connect to ESP32 GND |

**Wiring:**
```
Servo Signal → ESP32 GPIO 13
Servo VCC    → External 5V Power Supply (or ESP32 5V pin)
Servo GND    → ESP32 GND
```

#### Servo 3 - Pedestrian Gate
| Pin | Connection | Notes |
|-----|------------|-------|
| Signal (Orange/Yellow) | ESP32 GPIO 14 | PWM control |
| Power (Red) | External 5V (recommended) or ESP32 5V | Use external supply for stability |
| Ground (Black/Brown) | Common GND | Connect to ESP32 GND |

**Wiring:**
```
Servo Signal → ESP32 GPIO 14
Servo VCC    → External 5V Power Supply (or ESP32 5V pin)
Servo GND    → ESP32 GND
```

**⚠️ Important:** For multiple servos, use an external 5V 3A power supply and connect all servo VCC together, all GND together to common ground.

---

### Sensors

#### MQ135 Gas Sensor (Air Pollution)
| Pin | Connection | Notes |
|-----|------------|-------|
| VCC | ESP32 3.3V | Power supply |
| GND | ESP32 GND | Ground |
| AO (Analog Output) | ESP32 GPIO 34 (ADC1_CH6) | Analog signal |
| DO (Digital Output) | Not used | Leave disconnected |

**Wiring:**
```
MQ135 VCC → ESP32 3.3V
MQ135 GND → ESP32 GND
MQ135 AO  → ESP32 GPIO 34
MQ135 DO  → (Not connected)
```

---

#### IR Sensors (Speed Detection - 2 sensors needed)

**IR Sensor 1 (Entry point):**
| Pin | Connection | Notes |
|-----|------------|-------|
| VCC | ESP32 3.3V or 5V | Power supply |
| GND | ESP32 GND | Ground |
| OUT | ESP32 GPIO 26 | Digital signal (LOW when obstacle detected) |

**Wiring:**
```
IR Sensor 1 VCC → ESP32 3.3V (or 5V)
IR Sensor 1 GND → ESP32 GND
IR Sensor 1 OUT → ESP32 GPIO 26
```

**IR Sensor 2 (Exit point):**
| Pin | Connection | Notes |
|-----|------------|-------|
| VCC | ESP32 3.3V or 5V | Power supply |
| GND | ESP32 GND | Ground |
| OUT | ESP32 GPIO 27 | Digital signal (LOW when obstacle detected) |

**Wiring:**
```
IR Sensor 2 VCC → ESP32 3.3V (or 5V)
IR Sensor 2 GND → ESP32 GND
IR Sensor 2 OUT → ESP32 GPIO 27
```

**Note:** Position IR sensors 5 meters apart for speed calculation. Ensure they are aligned at the same height.

---

#### LDR Module (Light Dependent Resistor) - Day/Night Detection
| Module Pin | Connection | Notes |
|------------|------------|-------|
| VCC | ESP32 3.3V | Power supply |
| GND | ESP32 GND | Ground |
| DO (Digital Output) | ESP32 GPIO 35 (ADC1_CH7) | Digital signal (1 = light, 0 = dark) |
| AO (Analog Output) | Not used | Leave disconnected |

**Wiring:**
```
LDR Module VCC → ESP32 3.3V
LDR Module GND → ESP32 GND
LDR Module DO  → ESP32 GPIO 35
```

**Note:** LDR module has built-in resistance and voltage divider circuit. No external resistors needed.

---

#### HC-SR04 Ultrasonic Sensor (Vehicle Detection)
| Pin | Connection | Notes |
|-----|------------|-------|
| VCC | ESP32 5V | Power supply (5V required) |
| GND | ESP32 GND | Ground |
| Trig (Trigger) | ESP32 GPIO 32 | Trigger signal |
| Echo | ESP32 GPIO 33 | Echo signal (module is already 3.3V safe) |

**Wiring:**
```
HC-SR04 VCC  → ESP32 5V
HC-SR04 GND  → ESP32 GND
HC-SR04 Trig → ESP32 GPIO 32
HC-SR04 Echo → ESP32 GPIO 33
```

**Note:** Ultrasonic sensor module is already compatible with 3.3V logic. No voltage divider resistors needed.

---

#### Pedestrian Button
| Connection | Notes |
|------------|-------|
| One terminal | ESP32 GPIO 0 | Uses internal pull-up |
| Other terminal | ESP32 GND | Ground connection |

**Wiring:**
```
Button Terminal 1 → ESP32 GPIO 0
Button Terminal 2 → ESP32 GND

Note: Code uses INPUT_PULLUP mode, so button press = LOW signal
```

---

### I2C LCD Display (Direct Connection to ESP32)

| LCD Pin | ESP32 Pin | Notes |
|---------|-----------|-------|
| VCC | ESP32 5V or 3.3V | Power supply (check LCD module specs) |
| GND | ESP32 GND | Common ground |
| SDA | ESP32 GPIO 21 | I2C Data line |
| SCL | ESP32 GPIO 22 | I2C Clock line |

**Wiring:**
```
I2C LCD VCC → ESP32 5V (or 3.3V)
I2C LCD GND → ESP32 GND
I2C LCD SDA → ESP32 GPIO 21
I2C LCD SCL → ESP32 GPIO 22
```

**⚠️ Important:** 
- Most I2C LCD modules use address 0x27 or 0x3F (update in code if different)
- I2C lines have built-in pull-up resistors on the LCD module
- Keep I2C wires short (< 20cm recommended)
- Ensure common ground connection

---

### ESP32 Master Power Supply

| Pin | Connection | Notes |
|-----|------------|-------|
| VIN or 5V | 5V 2A adapter (shared with ESP32-CAM) | From common power adapter |
| GND | Common ground | All components share this |

**Power Configuration:**
- ESP32 Master: 5V 2A adapter (shared with ESP32-CAM)
- Alternative: Can use 7.4V battery pack with voltage regulator

---

## 📷 ESP32-CAM Connections

### Camera Module
The camera is **integrated** on the ESP32-CAM board. No external connections needed for the camera itself.

### Power Supply (Critical!)

| Pin | Connection | Notes |
|-----|------------|-------|
| 5V | External 5V 2A power supply | **Required!** USB may not be sufficient |
| GND | Common ground | Connect to power supply GND |

**⚠️ Critical:** ESP32-CAM requires stable 5V 2A power supply. USB connection may cause instability.

**Wiring:**
```
External 5V 2A Adapter → ESP32-CAM 5V pin
External Power GND      → ESP32-CAM GND pin
```

### Programming/Serial Communication (MB Module)

For uploading code, you need an MB (USB-to-Serial) module:

| MB Pin | ESP32-CAM Pin | Notes |
|--------|---------------|-------|
| TX | U0R (GPIO 3) | Serial receive |
| RX | U0T (GPIO 1) | Serial transmit |
| GND | GND | Common ground |
| 5V | 5V | Power (or use external supply) |
| DTR | GPIO 0 | Auto-reset for programming |

**⚠️ Important:** 
- **Hold BOOT button** on ESP32-CAM while clicking Upload
- Release BOOT button after upload starts
- GPIO 0 must be LOW for programming mode

**Wiring:**
```
MB Module TX  → ESP32-CAM GPIO 3
MB Module RX  → ESP32-CAM GPIO 1
MB Module GND → ESP32-CAM GND
MB Module 5V  → ESP32-CAM 5V (optional if using external supply)
MB Module DTR → ESP32-CAM GPIO 0 (for auto-reset)
```

---


## ⚡ Power Distribution

### Power Setup

Power configuration for the system:

1. **5V 2A Adapter** (Shared):
   - Powers ESP32 Master (VIN pin)
   - Powers ESP32-CAM (5V pin)
   - Both share the same 5V 2A adapter

2. **7.4V Battery Pack 1**:
   - Powers servos (via voltage regulator if needed)
   - Or powers ESP32 Master as alternative

3. **7.4V Battery Pack 2**:
   - Backup power or additional power source
   - Can power LCD or other components via regulator

### Common Ground Connection

**⚠️ CRITICAL:** All components must share a common ground!

```
ESP32 Master GND ←→ LCD GND
ESP32 Master GND ←→ Servo GND (all servos)
ESP32 Master GND ←→ Sensor GND (all sensors)
External Power GND ←→ ESP32-CAM GND
External Power GND ←→ ESP32 Master GND
```

### Power Distribution Diagram

```
5V 2A Adapter (Shared)
├── ESP32 Master VIN
└── ESP32-CAM 5V

7.4V Battery Pack 1
└── Servos (via regulator if needed)

7.4V Battery Pack 2
└── Backup/Additional Power

Common GND Bus
├── ESP32 Master GND
├── ESP32-CAM GND
├── LCD GND
├── All Servos GND
└── All Sensors GND
```

---

## 🔗 Component Interconnections

### System Communication Flow

```
Raspberry Pi (AI Vision)
    ↓ (WiFi)
Firebase Realtime Database
    ↑ (WiFi)
ESP32 Master (Controller)
    ↓ (I2C)
LCD Display (Direct Connection to ESP32)
```

### Physical Wire Connections Summary

1. **ESP32 Master ↔ LCD Display**: I2C (2 wires + power + GND)
   - SDA: GPIO 21 → LCD SDA
   - SCL: GPIO 22 → LCD SCL
   - VCC: 5V → LCD VCC
   - GND: ESP32 GND → LCD GND

2. **ESP32 Master ↔ Sensors**: Individual connections
   - Each sensor connects directly to ESP32 pins

3. **ESP32 Master ↔ Servos**: Individual PWM signals
   - Each servo signal wire to separate GPIO pin
   - Servos share external power supply

4. **ESP32 Master ↔ Traffic Light Module**: Direct GPIO connections
   - Single traffic light module with 4 pins (Red, Yellow, Green, GND)
   - Module has built-in resistors
   - Note: Other lanes are simulated in software

---

## 📐 Connection Diagrams

### ESP32 Master Pin Summary

| Pin Type | GPIO Pins Used | Purpose |
|----------|----------------|---------|
| Digital Output | 2, 4, 5 | Traffic light (3 LEDs: Red, Yellow, Green) |
| PWM Output | 12, 13, 14 | Servo motors (3 servos) |
| Analog Input | 34, 35 | MQ135, LDR sensors |
| Digital Input | 0, 26, 27, 33 | Button, IR sensors, Ultrasonic echo |
| Digital Output | 32 | Ultrasonic trigger |
| I2C | 21 (SDA), 22 (SCL) | LCD Display (I2C) |

### ESP32-CAM Pin Summary

| Pin | Purpose | Notes |
|-----|---------|-------|
| 5V | Power supply | External 5V 2A required |
| GND | Ground | Common ground |
| GPIO 1 (U0T) | Serial TX | For programming |
| GPIO 3 (U0R) | Serial RX | For programming |
| GPIO 0 | Boot mode | Hold LOW for programming |
| Camera pins | Integrated | Internal connections |

### I2C LCD Pin Summary

| Pin | Purpose | Connection |
|-----|---------|------------|
| SDA | I2C Data | ESP32 GPIO 21 |
| SCL | I2C Clock | ESP32 GPIO 22 |
| VCC | Power | ESP32 5V |
| GND | Ground | ESP32 GND (Common) |

---

## ⚠️ Important Safety Notes

1. **Power Requirements:**
   - Never exceed component voltage ratings
   - Use appropriate current ratings for power supplies
   - ESP32-CAM requires stable 5V 2A supply

2. **Signal Levels:**
   - ESP32 GPIO: 3.3V logic (max 3.3V input!)
   - All modules (LDR, Ultrasonic, Traffic Light) are compatible
   - I2C LCD: 5V or 3.3V compatible (check module specifications)

3. **Ground Connections:**
   - Always connect all GND together (common ground)
   - Serial communication requires common ground between devices

4. **Component Protection:**
   - Traffic light module has built-in resistors
   - All sensor modules are pre-configured for ESP32
   - Ensure proper power supply ratings (5V 2A for main system)

5. **Wiring Best Practices:**
   - Keep wires organized and labeled
   - Use appropriate wire gauge for power lines
   - Keep signal wires away from power wires
   - Use breadboard for prototyping

---

## 🔧 Component Shopping List

### Required Components:

- [ ] ESP32 Development Board × 1
- [ ] ESP32-CAM Module × 1
- [ ] I2C LCD 16×2 × 1
- [ ] MQ135 Gas Sensor × 1
- [ ] IR Sensors × 2
- [ ] LDR Module (Light Dependent Resistor) × 1 (with built-in resistance)
- [ ] HC-SR04 Ultrasonic Sensor × 1 (3.3V compatible)
- [ ] Servo Motors × 3
- [ ] Traffic Light Module × 1 (with built-in resistors, 4 pins: Red, Yellow, Green, GND)
- [ ] Push Button × 1
- [ ] Jumper Wires (multiple)
- [ ] Breadboard (for prototyping)
- [ ] Power Supplies:
  - 5V 2A adapter × 1 (shared for ESP32 Master and ESP32-CAM)
  - 7.4V Battery Pack × 2 (for servos and backup power)
- [ ] MB Module (USB-to-Serial) × 1 (for ESP32-CAM programming)

---

## 📝 Quick Connection Checklist

Before powering on, verify:

- [ ] All power supplies are correct voltage and current
- [ ] All GND connections are connected together
- [ ] Traffic light module has built-in resistors (no external resistors needed)
- [ ] I2C LCD connection: ESP32 GPIO 21 (SDA) → LCD SDA, GPIO 22 (SCL) → LCD SCL
- [ ] Ultrasonic sensor is 3.3V compatible (no voltage divider needed)
- [ ] LDR module has built-in circuit (no external resistors needed)
- [ ] Servos have external power (if using multiple)
- [ ] All devices share common ground
- [ ] No short circuits between power and ground
- [ ] All connections are secure and properly soldered/crimped

---

## 🐛 Troubleshooting Common Issues

**ESP32 won't connect to WiFi:**
- Check WiFi credentials in code
- Ensure 2.4GHz network (ESP32 doesn't support 5GHz)
- Check signal strength

**LCD not displaying:**
- Verify I2C connections (SDA → GPIO 21, SCL → GPIO 22)
- Check I2C address matches in code (usually 0x27 or 0x3F)
- Use I2C scanner to find correct LCD address
- Ensure common ground between ESP32 and LCD
- Check LCD power supply (5V or 3.3V depending on module)

**Servo not moving:**
- Check power supply (may need external 5V)
- Verify signal pin connection
- Check code servo pin assignment

**Sensors not reading correctly:**
- Verify pin connections in code
- Check power supply to sensors
- Verify analog/digital pin types match code

**ESP32-CAM not working:**
- Ensure stable 5V 2A power supply
- Check camera initialization in code
- Verify camera module is properly connected

---

**Complete all connections before powering on the system!** 🔌✅

