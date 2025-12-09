#include <WiFi.h>
#include <FirebaseESP32.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ============ WiFi & Firebase Configuration ============
#define WIFI_SSID "H&M"
#define WIFI_PASSWORD "123456798"
#define FIREBASE_HOST "https://smart-traffic-system-4ac4b-default-rtdb.firebaseio.com/"
#define FIREBASE_AUTH "AIzaSyB654P2Pdrx7EUkD1RmLFIZq5jFo2RAki4"

FirebaseData firebaseData;
FirebaseJson json;

// ============ Pin Definitions ============
#define RED_PIN 2
#define YELLOW_PIN 4
#define GREEN_PIN 5

// Servos
Servo emergencyGateServo;
Servo speedBumpServo;
Servo pedestrianGateServo;
#define SERVO_EMERGENCY_PIN 12
#define SERVO_SPEEDBUMP_PIN 13
#define SERVO_PEDESTRIAN_PIN 14

// Street Lights Relay (controls all 10 white LEDs)
#define STREET_LIGHTS_RELAY_PIN 15

// Sensors
#define MQ135_PIN 34           // Air pollution sensor (Analog)
#define IR_SENSOR_1_PIN 26     // Speed detection sensor 1
#define IR_SENSOR_2_PIN 27     // Speed detection sensor 2
#define LDR_PIN 35             // Day/Night sensor (Analog)
#define ULTRASONIC_TRIG 32     // Ultrasonic trigger
#define ULTRASONIC_ECHO 33     // Ultrasonic echo
#define PEDESTRIAN_BUTTON 0    // Pedestrian crossing button

// I2C LCD Display (Direct Connection)
#define LCD_I2C_ADDRESS 0x27  // I2C address (0x27 or 0x3F - check your LCD module)
#define LCD_COLUMNS 16
#define LCD_ROWS 2
LiquidCrystal_I2C lcd(LCD_I2C_ADDRESS, LCD_COLUMNS, LCD_ROWS);

// ============ Global Variables ============
bool emergencyMode = false;
bool failSafeMode = false;
unsigned long lastFirebaseCheck = 0;
unsigned long lastTrafficUpdate = 0;
unsigned long lastSpeedCheck = 0;
unsigned long lastPollutionRead = 0;

// Traffic Light States (Single physical light - other lanes simulated)
enum TrafficState {
  RED,
  YELLOW,
  GREEN
};

TrafficState currentTrafficState = RED;
TrafficState lane1State = RED;  // Simulated
TrafficState lane2State = RED;  // Simulated
TrafficState lane3State = RED;  // Simulated

// Timing (will be adjusted by AI)
int lane1GreenTime = 5000;  // Default 5 seconds
int lane2GreenTime = 5000;
int lane3GreenTime = 5000;
int yellowTime = 2000;      // 2 seconds yellow

// Sensor Values
float pollutionPPM = 0;
bool isDayTime = true;
bool vehicleDetected = false;
unsigned long lastVehicleDetectTime = 0;
int speedViolations = 0;

// Speed Detection
unsigned long timeSensor1 = 0;
unsigned long timeSensor2 = 0;
float vehicleSpeed = 0;
const float distanceBetweenSensors = 5.0; // meters

void setup() {
  Serial.begin(115200);
  
  // Initialize I2C LCD Display
  Wire.begin();
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Smart Road Eye");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");
  
  // Initialize pins
  initializePins();
  
  // Initialize servos
  initializeServos();
  
  // Connect to WiFi
  connectWiFi();
  
  // Initialize Firebase
  initializeFirebase();
  
  // Set initial states
  setInitialStates();
  
  Serial.println("ESP32 Master Controller Initialized!");
  sendToLCD("System Ready");
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Check Firebase connection every 5 seconds
  if (currentMillis - lastFirebaseCheck > 5000) {
    checkFirebaseConnection();
    lastFirebaseCheck = currentMillis;
  }
  
  // Read sensors
  readSensors();
  
  // Check for emergency mode
  checkEmergencyMode();
  
  // Process traffic lights (only if not in emergency)
  if (!emergencyMode) {
    processTrafficLights(currentMillis);
    processPedestrianRequest();
  } else {
    handleEmergencyMode();
  }
  
  // Check speed violations
  if (currentMillis - lastSpeedCheck > 1000) {
    checkSpeedViolations();
    lastSpeedCheck = currentMillis;
  }
  
  // Update Firebase every 2 seconds
  if (currentMillis - lastTrafficUpdate > 2000) {
    updateFirebase();
    lastTrafficUpdate = currentMillis;
  }
  
  // Read from Firebase (AI commands, emergency flags)
  readFirebaseCommands();
  
  delay(100);
}

// ============ Initialization Functions ============
void initializePins() {
  // Traffic light (single physical light)
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  
  // Sensors
  pinMode(IR_SENSOR_1_PIN, INPUT);
  pinMode(IR_SENSOR_2_PIN, INPUT);
  pinMode(LDR_PIN, INPUT);
  pinMode(ULTRASONIC_TRIG, OUTPUT);
  pinMode(ULTRASONIC_ECHO, INPUT);
  pinMode(PEDESTRIAN_BUTTON, INPUT_PULLUP);
  
  // Street Lights Relay (controls all 10 white LEDs)
  pinMode(STREET_LIGHTS_RELAY_PIN, OUTPUT);
  digitalWrite(STREET_LIGHTS_RELAY_PIN, LOW); // Initially off (relay LOW = lights off)
  
  // Set physical traffic light to red initially
  digitalWrite(RED_PIN, HIGH);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
  currentTrafficState = RED;
}

void initializeServos() {
  emergencyGateServo.attach(SERVO_EMERGENCY_PIN);
  speedBumpServo.attach(SERVO_SPEEDBUMP_PIN);
  pedestrianGateServo.attach(SERVO_PEDESTRIAN_PIN);
  
  // Initial positions (closed)
  emergencyGateServo.write(0);      // Gate closed
  speedBumpServo.write(0);          // Bump down
  pedestrianGateServo.write(0);     // Gate closed
}

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi Connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi Connection Failed!");
    failSafeMode = true;
  }
}

void initializeFirebase() {
  Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);
  Firebase.reconnectWiFi(true);
  Firebase.setReadTimeout(firebaseData, 1000 * 60);
  Firebase.setwriteSizeLimit(firebaseData, "tiny");
  
  // Test connection
  if (Firebase.setInt(firebaseData, "/system/esp32_status", 1)) {
    Serial.println("Firebase Connected!");
    failSafeMode = false;
  } else {
    Serial.println("Firebase Connection Failed!");
    failSafeMode = true;
  }
}

void setInitialStates() {
  lane1State = RED;
  lane2State = RED;
  lane3State = RED;
  emergencyMode = false;
  
  // Close all gates
  emergencyGateServo.write(0);
  speedBumpServo.write(0);
  pedestrianGateServo.write(0);
}

// ============ Sensor Reading Functions ============
void readSensors() {
  // Read MQ135 (Pollution)
  int mq135Value = analogRead(MQ135_PIN);
  // Convert to PPM (calibration needed for your sensor)
  pollutionPPM = map(mq135Value, 0, 4095, 0, 1000); // Approximate conversion
  
  // Read LDR (Day/Night)
  int ldrValue = analogRead(LDR_PIN);
  isDayTime = (ldrValue > 512); // Threshold may need adjustment
  
  // Control Street Lights based on day/night
  controlStreetLights();
  
  // Read Ultrasonic (Vehicle Detection)
  float distance = readUltrasonic();
  vehicleDetected = (distance < 50 && distance > 0); // Vehicle within 50cm
  
  if (vehicleDetected) {
    lastVehicleDetectTime = millis();
  }
}

float readUltrasonic() {
  digitalWrite(ULTRASONIC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG, LOW);
  
  long duration = pulseIn(ULTRASONIC_ECHO, HIGH, 30000);
  float distance = (duration * 0.034) / 2; // Convert to cm
  
  return distance;
}

// ============ Street Lights Control ============
void controlStreetLights() {
  // Turn on street lights (all 10 white LEDs) when it's dark (night time)
  // Turn off when it's bright (day time)
  if (!isDayTime) {
    // Night time - turn on street lights
    digitalWrite(STREET_LIGHTS_RELAY_PIN, HIGH); // Relay HIGH = lights ON
  } else {
    // Day time - turn off street lights
    digitalWrite(STREET_LIGHTS_RELAY_PIN, LOW); // Relay LOW = lights OFF
  }
}

// ============ Traffic Light Control ============
void processTrafficLights(unsigned long currentMillis) {
  static unsigned long lane1Timer = 0;
  static unsigned long lane2Timer = 0;
  static unsigned long lane3Timer = 0;
  static int currentLane = 1;
  static unsigned long lastLCDUpdate = 0;
  
  // Cycle through lanes
  switch(currentLane) {
    case 1:
      if (lane1State == GREEN && (currentMillis - lane1Timer) > lane1GreenTime) {
        lane1State = YELLOW;
        updatePhysicalTrafficLight(YELLOW);  // Update physical light
        lane1Timer = currentMillis;
      } else if (lane1State == YELLOW && (currentMillis - lane1Timer) > yellowTime) {
        setLaneState(1, RED);
        lane1State = RED;
        currentLane = 2;
        lane2State = GREEN;
        setLaneState(2, GREEN);
        updatePhysicalTrafficLight(GREEN);  // Update physical light
        lane2Timer = currentMillis;
      } else if (lane1State == GREEN) {
        updatePhysicalTrafficLight(GREEN);  // Update physical light for active lane
      } else if (lane1State == YELLOW) {
        updatePhysicalTrafficLight(YELLOW);  // Update physical light for active lane
      }
      // Update LCD with remaining time for current lane
      if (currentMillis - lastLCDUpdate > 1000) { // Update every second
        updateLCDTrafficTime(currentLane, currentMillis, lane1Timer, lane1GreenTime, yellowTime);
        lastLCDUpdate = currentMillis;
      }
      break;
      
    case 2:
      if (lane2State == GREEN && (currentMillis - lane2Timer) > lane2GreenTime) {
        lane2State = YELLOW;
        updatePhysicalTrafficLight(YELLOW);  // Update physical light
        lane2Timer = currentMillis;
      } else if (lane2State == YELLOW && (currentMillis - lane2Timer) > yellowTime) {
        setLaneState(2, RED);
        lane2State = RED;
        currentLane = 3;
        lane3State = GREEN;
        setLaneState(3, GREEN);
        updatePhysicalTrafficLight(GREEN);  // Update physical light
        lane3Timer = currentMillis;
      } else if (lane2State == GREEN) {
        updatePhysicalTrafficLight(GREEN);  // Update physical light for active lane
      } else if (lane2State == YELLOW) {
        updatePhysicalTrafficLight(YELLOW);  // Update physical light for active lane
      }
      if (currentMillis - lastLCDUpdate > 1000) {
        updateLCDTrafficTime(currentLane, currentMillis, lane2Timer, lane2GreenTime, yellowTime);
        lastLCDUpdate = currentMillis;
      }
      break;
      
    case 3:
      if (lane3State == GREEN && (currentMillis - lane3Timer) > lane3GreenTime) {
        lane3State = YELLOW;
        updatePhysicalTrafficLight(YELLOW);  // Update physical light
        lane3Timer = currentMillis;
      } else if (lane3State == YELLOW && (currentMillis - lane3Timer) > yellowTime) {
        setLaneState(3, RED);
        lane3State = RED;
        currentLane = 1;
        lane1State = GREEN;
        setLaneState(1, GREEN);
        updatePhysicalTrafficLight(GREEN);  // Update physical light
        lane1Timer = currentMillis;
      } else if (lane3State == GREEN) {
        updatePhysicalTrafficLight(GREEN);  // Update physical light for active lane
      } else if (lane3State == YELLOW) {
        updatePhysicalTrafficLight(YELLOW);  // Update physical light for active lane
      }
      if (currentMillis - lastLCDUpdate > 1000) {
        updateLCDTrafficTime(currentLane, currentMillis, lane3Timer, lane3GreenTime, yellowTime);
        lastLCDUpdate = currentMillis;
      }
      break;
  }
}

// Calculate and send remaining time to LCD
void updateLCDTrafficTime(int lane, unsigned long currentMillis, unsigned long laneTimer, int greenTime, int yellowTime) {
  String stateStr = "";
  int remainingTime = 0;
  
  if (lane == 1) {
    if (lane1State == GREEN) {
      stateStr = "GREEN";
      remainingTime = (greenTime - (currentMillis - laneTimer)) / 1000;
    } else if (lane1State == YELLOW) {
      stateStr = "YELLOW";
      remainingTime = (yellowTime - (currentMillis - laneTimer)) / 1000;
    } else {
      stateStr = "RED";
      remainingTime = 0;
    }
  } else if (lane == 2) {
    if (lane2State == GREEN) {
      stateStr = "GREEN";
      remainingTime = (greenTime - (currentMillis - laneTimer)) / 1000;
    } else if (lane2State == YELLOW) {
      stateStr = "YELLOW";
      remainingTime = (yellowTime - (currentMillis - laneTimer)) / 1000;
    } else {
      stateStr = "RED";
      remainingTime = 0;
    }
  } else if (lane == 3) {
    if (lane3State == GREEN) {
      stateStr = "GREEN";
      remainingTime = (greenTime - (currentMillis - laneTimer)) / 1000;
    } else if (lane3State == YELLOW) {
      stateStr = "YELLOW";
      remainingTime = (yellowTime - (currentMillis - laneTimer)) / 1000;
    } else {
      stateStr = "RED";
      remainingTime = 0;
    }
  }
  
  // Display on I2C LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Lane " + String(lane) + ": " + stateStr);
  lcd.setCursor(0, 1);
  lcd.print("Time: " + String(remainingTime) + "s");
}

void setLaneState(int lane, TrafficState state) {
  // Update simulated lane state for Firebase/app
  switch(lane) {
    case 1:
      lane1State = state;
      break;
    case 2:
      lane2State = state;
      break;
    case 3:
      lane3State = state;
      break;
    default:
      return;
  }
  
  // Only update physical traffic light if this is the active lane
  // Physical light shows the state of the currently active lane
  updatePhysicalTrafficLight(state);
}

void updatePhysicalTrafficLight(TrafficState state) {
  // Turn off all lights first
  digitalWrite(RED_PIN, LOW);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
  
  // Turn on appropriate light
  switch(state) {
    case RED:
      digitalWrite(RED_PIN, HIGH);
      currentTrafficState = RED;
      break;
    case YELLOW:
      digitalWrite(YELLOW_PIN, HIGH);
      currentTrafficState = YELLOW;
      break;
    case GREEN:
      digitalWrite(GREEN_PIN, HIGH);
      currentTrafficState = GREEN;
      break;
  }
}

// ============ Emergency Mode ============
void checkEmergencyMode() {
  if (Firebase.getInt(firebaseData, "/system/emergency_mode")) {
    emergencyMode = (firebaseData.intData() == 1);
  }
}

void handleEmergencyMode() {
  // Get emergency direction from Firebase
  int emergencyDirection = 1;
  if (Firebase.getInt(firebaseData, "/system/emergency_direction")) {
    emergencyDirection = firebaseData.intData();
  }
  
  // Set all lanes to green for emergency direction (simulated)
  // Set other lanes to red (simulated)
  for (int i = 1; i <= 3; i++) {
    if (i == emergencyDirection) {
      setLaneState(i, GREEN);
      updatePhysicalTrafficLight(GREEN);  // Physical light shows emergency (green)
    } else {
      setLaneState(i, RED);
    }
  }
  
  // Open emergency gate
  emergencyGateServo.write(90);
  
  // Close pedestrian gate (safety)
  pedestrianGateServo.write(0);
  
  // Send emergency alert to LCD
  sendToLCD("NOTE:EMERGENCY:Yield");
}

// ============ Pedestrian System ============
void processPedestrianRequest() {
  static bool lastButtonState = HIGH;
  static unsigned long lastDebounceTime = 0;
  static bool pedestrianActive = false;
  static unsigned long pedestrianTimer = 0;
  
  bool buttonState = digitalRead(PEDESTRIAN_BUTTON);
  
  // Button pressed (LOW because of pull-up)
  if (buttonState == LOW && lastButtonState == HIGH && !emergencyMode) {
    if (millis() - lastDebounceTime > 50) {
      pedestrianActive = true;
      pedestrianTimer = millis();
      openPedestrianGate();
      sendToLCD("NOTE:PEDESTRIAN:Crossing");
    }
    lastDebounceTime = millis();
  }
  
  lastButtonState = buttonState;
  
  // Close gate after 10 seconds
  if (pedestrianActive && (millis() - pedestrianTimer > 10000)) {
    closePedestrianGate();
    pedestrianActive = false;
    sendToLCD("NOTE:CLEAR:Continue");
  }
}

void openPedestrianGate() {
  pedestrianGateServo.write(90);
}

void closePedestrianGate() {
  pedestrianGateServo.write(0);
}

// ============ Speed Detection ============
void checkSpeedViolations() {
  static bool sensor1Triggered = false;
  static bool sensor2Triggered = false;
  
  bool sensor1 = !digitalRead(IR_SENSOR_1_PIN);
  bool sensor2 = !digitalRead(IR_SENSOR_2_PIN);
  
  if (sensor1 && !sensor1Triggered) {
    timeSensor1 = millis();
    sensor1Triggered = true;
  }
  
  if (sensor2 && !sensor2Triggered && sensor1Triggered) {
    timeSensor2 = millis();
    sensor2Triggered = true;
    
    // Calculate speed
    unsigned long timeDifference = timeSensor2 - timeSensor1;
    if (timeDifference > 0) {
      vehicleSpeed = (distanceBetweenSensors * 1000.0) / timeDifference; // m/s
      vehicleSpeed = vehicleSpeed * 3.6; // Convert to km/h
      
      // Check if speeding (assuming speed limit is 50 km/h)
      if (vehicleSpeed > 50) {
        speedViolations++;
        raiseSpeedBump();
        
        // Log violation to Firebase
        Firebase.pushInt(firebaseData, "/violations/speed", (int)vehicleSpeed);
        
        sendToLCD("SPEED VIOLATION!");
      }
    }
    
    // Reset
    sensor1Triggered = false;
    sensor2Triggered = false;
  }
  
  // Reset after timeout
  if (sensor1Triggered && (millis() - timeSensor1 > 5000)) {
    sensor1Triggered = false;
  }
}

void raiseSpeedBump() {
  speedBumpServo.write(90);  // Raise bump
  sendToLCD("NOTE:SPEED:Violation");
  delay(3000);                // Keep raised for 3 seconds
  speedBumpServo.write(0);    // Lower bump
}

// ============ Firebase Communication ============
void updateFirebase() {
  if (failSafeMode) return;
  
  // Update traffic light states
  json.set("/traffic/lane1", (lane1State == GREEN) ? "green" : (lane1State == YELLOW) ? "yellow" : "red");
  json.set("/traffic/lane2", (lane2State == GREEN) ? "green" : (lane2State == YELLOW) ? "yellow" : "red");
  json.set("/traffic/lane3", (lane3State == GREEN) ? "green" : (lane3State == YELLOW) ? "yellow" : "red");
  
  // Update sensors
  json.set("/sensors/pollution_ppm", pollutionPPM);
  json.set("/sensors/daytime", isDayTime);
  json.set("/sensors/vehicle_detected", vehicleDetected);
  
  // Update system status
  json.set("/system/emergency_mode", emergencyMode ? 1 : 0);
  json.set("/system/failsafe_mode", failSafeMode ? 1 : 0);
  json.set("/system/speed_violations", speedViolations);
  
  // Update timing (from AI)
  if (Firebase.getInt(firebaseData, "/ai/lane1_green_time")) {
    lane1GreenTime = firebaseData.intData();
  }
  if (Firebase.getInt(firebaseData, "/ai/lane2_green_time")) {
    lane2GreenTime = firebaseData.intData();
  }
  if (Firebase.getInt(firebaseData, "/ai/lane3_green_time")) {
    lane3GreenTime = firebaseData.intData();
  }
  
  Firebase.updateNode(firebaseData, "/road_status", json);
}

void readFirebaseCommands() {
  // Emergency mode already checked in checkEmergencyMode()
  // AI timing already updated in updateFirebase()
}

void checkFirebaseConnection() {
  if (Firebase.getInt(firebaseData, "/system/esp32_status")) {
    failSafeMode = false;
  } else {
    failSafeMode = true;
    activateFailSafe();
  }
}

void activateFailSafe() {
  // Default traffic cycle
  lane1State = RED;
  lane2State = RED;
  lane3State = RED;
  
  // Close all gates
  emergencyGateServo.write(0);
  speedBumpServo.write(0);
  pedestrianGateServo.write(0);
  
  // Send to LCD
  sendToLCD("NOTE:FAILSAFE:Active");
}

// ============ I2C LCD Display Functions ============
void sendToLCD(String message) {
  lcd.clear();
  lcd.setCursor(0, 0);
  
  // Split message if too long (max 16 chars per line)
  if (message.length() <= 16) {
    lcd.print(message);
  } else {
    // Split into two lines if message is longer than 16 characters
    lcd.print(message.substring(0, 16));
    lcd.setCursor(0, 1);
    lcd.print(message.substring(16, 32));
  }
}

