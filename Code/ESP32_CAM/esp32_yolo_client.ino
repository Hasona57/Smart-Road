/*
 * ESP32-CAM YOLO Detection Client
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
#include <ArduinoJson.h>

// ========== CONFIGURATION ==========
const char* ssid = "H&M";           // Change this!
const char* password = "123456798";    // Change this!

// Your laptop's IP address (run `ipconfig` on Windows to find it)
// Example: "192.168.1.100"
String serverUrl = "http://192.168.8.238:5000/detect";  // Change this!

// Camera frame rate (milliseconds between captures)
const int captureInterval = 300;  // 300ms = ~3 FPS (good balance for quality)

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
  Serial.println("ESP32-CAM YOLO Detection Client");
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
  
  // Keep good quality for accuracy - QVGA or VGA
  config.frame_size = FRAMESIZE_QVGA;  // 320x240 - good balance of quality and speed
  // For better accuracy use: FRAMESIZE_VGA (640x480)
  config.jpeg_quality = 10;  // Lower = better quality, but still reasonable file size
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
  // Ensure WiFi is connected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ WiFi disconnected! Reconnecting...");
    WiFi.disconnect();
    delay(100);
    WiFi.begin(ssid, password);
    
    int reconnectAttempts = 0;
    while (WiFi.status() != WL_CONNECTED && reconnectAttempts < 30) {
      delay(500);
      Serial.print(".");
      reconnectAttempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n✅ WiFi reconnected!");
      Serial.print("📡 IP: ");
      Serial.println(WiFi.localIP());
    } else {
      Serial.println("\n❌ WiFi reconnection failed!");
      delay(1000);
      return;
    }
  }
  
  // Capture frame
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("⚠️  Camera capture failed");
    delay(captureInterval);
    return;
  }
  
  // Reduced logging for speed - only log every 10 frames
  static int frameCount = 0;
  static int successCount = 0;
  static int failCount = 0;
  frameCount++;
  
  // Send to server with improved reliability
  HTTPClient http;
  bool success = false;
  int retryCount = 0;
  const int maxRetries = 3;
  
  while (!success && retryCount < maxRetries) {
    // Create fresh connection each time for reliability
    http.begin(serverUrl);
    http.addHeader("Content-Type", "image/jpeg");
    http.addHeader("Connection", "close");  // Use close for reliability
    http.setTimeout(5000);  // 5 second timeout
    http.setReuse(false);  // Don't reuse - create fresh connection
    
    unsigned long startTime = millis();
    int httpResponseCode = http.POST(fb->buf, fb->len);
    unsigned long requestTime = millis() - startTime;
    
    if (httpResponseCode > 0) {
      if (httpResponseCode == 200) {
        // Success - read response quickly
        String response = http.getString();
        success = true;
        successCount++;
        failCount = 0;  // Reset fail counter on success
        
        if (frameCount % 10 == 0) {
          Serial.printf("✅ Frame %d: OK (%lums, %d bytes)\n", frameCount, requestTime, fb->len);
        }
      } else {
        // Server error (not connection error)
        Serial.printf("⚠️  Server error %d\n", httpResponseCode);
        String response = http.getString();
        success = true;  // Don't retry on server errors
      }
    } else {
      // Connection error
      failCount++;
      retryCount++;
      
      if (retryCount < maxRetries) {
        // Exponential backoff: 100ms, 200ms, 400ms
        int backoffDelay = 100 * (1 << (retryCount - 1));
        Serial.printf("❌ Failed: %d (attempt %d/%d) - retrying in %dms...\n", 
                      httpResponseCode, retryCount, maxRetries, backoffDelay);
        delay(backoffDelay);
        
        // Check WiFi again before retry
        if (WiFi.status() != WL_CONNECTED) {
          Serial.println("⚠️  WiFi lost during retry - reconnecting...");
          WiFi.reconnect();
          delay(500);
        }
      } else {
        Serial.printf("❌ Failed: %d after %d attempts\n", httpResponseCode, maxRetries);
        
        // If too many consecutive failures, check server reachability
        if (failCount > 10) {
          Serial.println("⚠️  Multiple failures - checking server...");
          HTTPClient testHttp;
          String healthUrl = serverUrl;
          healthUrl.replace("/detect", "/health");
          testHttp.begin(healthUrl);
          testHttp.setTimeout(2000);
          int testCode = testHttp.GET();
          testHttp.end();
          
          if (testCode > 0) {
            Serial.println("✅ Server is reachable - continuing...");
            failCount = 0;  // Reset counter
          } else {
            Serial.println("❌ Server unreachable - check IP and server status");
            delay(2000);  // Wait longer before next attempt
          }
        }
      }
    }
    
    http.end();  // Always close connection
  }
  
  // Return frame buffer
  esp_camera_fb_return(fb);
  
  // Wait before next capture
  delay(captureInterval);
}

