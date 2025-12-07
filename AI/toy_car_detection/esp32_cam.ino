/*
 * ESP32-CAM Toy Car Detection Client
 * Sends images to Flask server for YOLO detection
 * 
 * Hardware: AI Thinker ESP32-CAM
 * 
 * Setup:
 * 1. Install ESP32 board support in Arduino IDE
 * 2. Install libraries: esp_camera, WiFi, HTTPClient
 * 3. Update WiFi credentials and server IP below
 * 4. Select board: AI Thinker ESP32-CAM
 * 5. Upload this sketch
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ========== CONFIGURATION ==========
const char* ssid = "H&M";           // Change this!
const char* password = "123456798";    // Change this!

// Your laptop's IP address (run `ipconfig` on Windows to find it)
// Example: "192.168.1.100"
String serverUrl = "http://192.168.8.238:5000/detect";  // Change this!

// Camera frame rate (milliseconds between captures)
const int captureInterval = 1000;  // 1 second = 1 FPS

// ========== CAMERA PINS (AI Thinker ESP32-CAM) ==========
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(2000);
  delay(1000);
  
  Serial.println("\n\n=================================");
  Serial.println("ESP32-CAM Toy Car Detection");
  Serial.println("=================================\n");
  
  // Initialize camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // Frame size: QVGA (320x240) for speed, or VGA (640x480) for quality
  config.frame_size = FRAMESIZE_QVGA;  // Change to FRAMESIZE_VGA for better quality
  config.jpeg_quality = 12;  // 0-63, lower = better quality but larger
  config.fb_count = 1;
  
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed with error 0x%x\n", err);
    return;
  }
  Serial.println("✅ Camera initialized");
  
  // Connect to WiFi
  Serial.print("📡 Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected!");
    Serial.print("📡 IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("📡 Server URL: ");
    Serial.println(serverUrl);
  } else {
    Serial.println("\n❌ WiFi connection failed!");
    return;
  }
  
  Serial.println("\n🚀 Ready! Starting detection loop...\n");
}

void loop() {
  // Capture frame
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("⚠️  Camera capture failed");
    delay(captureInterval);
    return;
  }
  
  Serial.print("📸 Captured image: ");
  Serial.print(fb->len);
  Serial.println(" bytes");
  
  // Send to server
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "image/jpeg");
    
    int httpResponseCode = http.POST(fb->buf, fb->len);
    
    if (httpResponseCode > 0) {
      Serial.print("✅ Response code: ");
      Serial.println(httpResponseCode);
      
      if (httpResponseCode == 200) {
        String response = http.getString();
        Serial.println("📦 Detections:");
        Serial.println(response);
        Serial.println();
      } else {
        Serial.print("⚠️  Server error: ");
        Serial.println(httpResponseCode);
      }
    } else {
      Serial.print("❌ HTTP request failed: ");
      Serial.println(httpResponseCode);
      Serial.println("💡 Check if server is running and IP is correct");
    }
    
    http.end();
  } else {
    Serial.println("❌ WiFi disconnected!");
    WiFi.reconnect();
  }
  
  // Return frame buffer
  esp_camera_fb_return(fb);
  
  // Wait before next capture
  delay(captureInterval);
}


