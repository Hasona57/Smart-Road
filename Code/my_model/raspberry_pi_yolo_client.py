#!/usr/bin/env python3
"""
Raspberry Pi YOLO Detection Client - Runs YOLO Locally
Real-time object detection using Raspberry Pi Camera Module

Hardware: Raspberry Pi 4 + Raspberry Pi Camera Module v2

This version runs YOLO directly on the Raspberry Pi for:
- Lower latency (no network overhead)
- Better performance
- No dependency on external server

Setup:
1. Install dependencies: sudo apt install python3-picamera2 python3-pip
2. Install Python packages: pip3 install -r requirements.txt
3. Enable camera: sudo raspi-config > Interface Options > Camera > Enable
4. Update model path and settings below
5. Run: python3 raspberry_pi_yolo_client.py
"""

import time
import cv2
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO
from picamera2 import Picamera2
import pyrebase

# ========== CONFIGURATION ==========
MODEL_PATH = "my_model.pt"  # Path to your YOLO model
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence threshold (0.0 to 1.0)
CAPTURE_INTERVAL = 0.3  # 300ms = ~3 FPS (time between captures)

# Camera settings
CAMERA_WIDTH = 640   # Image width in pixels
CAMERA_HEIGHT = 480  # Image height in pixels

# Display settings
SHOW_DISPLAY = True  # Set to False for headless operation (no screen needed)

# ========== Firebase Configuration ==========
# Same database as ESP32 Master
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyB654P2Pdrx7EUkD1RmLFIZq5jFo2RAki4",
    "authDomain": "smart-traffic-system-4ac4b.firebaseapp.com",
    "databaseURL": "https://smart-traffic-system-4ac4b-default-rtdb.firebaseio.com/",
    "storageBucket": "smart-traffic-system-4ac4b.appspot.com"
}

# Firebase update settings
SEND_TO_FIREBASE = True  # Set to False to disable Firebase
FIREBASE_UPDATE_INTERVAL = 1.0  # Update Firebase every 1 second (if detections exist)

# ========== Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_camera():
    """Initialize Raspberry Pi camera"""
    try:
        picam2 = Picamera2()
        
        # Configure camera
        camera_config = picam2.create_still_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
            lores={"size": (320, 240)},
            display="lores"
        )
        picam2.configure(camera_config)
        
        # Set camera controls
        picam2.set_controls({
            "ExposureTime": 10000,  # 10ms
            "AnalogueGain": 1.0,
            "AwbEnable": True,  # Auto white balance
        })
        
        picam2.start()
        time.sleep(2)  # Allow camera to stabilize
        logger.info("Camera initialized")
        return picam2
    except Exception as e:
        logger.error(f"Camera initialization failed: {e}")
        logger.error("Make sure camera is enabled: sudo raspi-config")
        return None


def load_yolo_model(model_path):
    """Load YOLO model"""
    try:
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        logger.info(f"Loading YOLO model from {model_path}...")
        model = YOLO(model_path, task='detect')
        labels = model.names
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"   Classes: {len(labels)}")
        logger.info(f"   Class names: {list(labels.values())[:10]}{'...' if len(labels) > 10 else ''}")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def capture_image_picam2(picam2):
    """Capture image from Pi Camera"""
    try:
        image_array = picam2.capture_array("main")
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception as e:
        logger.error(f"Camera capture failed: {e}")
        return None


def initialize_firebase():
    """Initialize Firebase connection"""
    try:
        firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
        db = firebase.database()
        
        # Test connection by reading a value
        try:
            db.child("system").child("raspberry_pi_status").get()
            logger.info("Firebase initialized and connected")
        except Exception as e:
            logger.warning(f"Firebase connected but test read failed: {e}")
            logger.warning("Continuing - writes may still work...")
        
        return db
    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")
        logger.warning("Continuing without Firebase...")
        logger.warning("Set SEND_TO_FIREBASE = False to disable Firebase messages")
        return None

def send_detections_to_firebase(db, detections, frame_count, inference_time):
    """Send detection results to Firebase Realtime Database"""
    if not db or not SEND_TO_FIREBASE:
        return
    
    try:
        # Prepare detection data
        detection_data = {
            'timestamp': int(time.time() * 1000),  # Milliseconds since epoch
            'frame_count': frame_count,
            'inference_time_ms': round(inference_time, 2),
            'detection_count': len(detections),
            'detections': detections
        }
        
        # Count objects by class
        class_counts = {}
        for det in detections:
            class_name = det['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        detection_data['class_counts'] = class_counts
        
        # Update Firebase - store in /ai/detections/latest
        db.child("ai").child("detections").child("latest").set(detection_data)
        
        # Also push to history (keeps last 100 entries)
        db.child("ai").child("detections").child("history").push(detection_data)
        
        # Update summary statistics
        summary = {
            'total_detections': len(detections),
            'last_update': int(time.time() * 1000),
            'classes_detected': list(class_counts.keys()),
            'class_counts': class_counts
        }
        db.child("ai").child("detections").child("summary").set(summary)
        
        logger.debug(f"Sent {len(detections)} detections to Firebase")
        
    except Exception as e:
        logger.error(f"Failed to send to Firebase: {e}")

def run_detection(model, image):
    """Run YOLO detection on image"""
    try:
        # Run inference
        results = model(
            image,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=640,
            verbose=False
        )
        
        # Process results
        detections = results[0].boxes
        detection_list = []
        
        if len(detections) > 0:
            for i in range(len(detections)):
                xyxy = detections[i].xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = xyxy.astype(int).tolist()
                
                class_idx = int(detections[i].cls.item())
                class_name = model.names[class_idx]
                confidence = float(detections[i].conf.item())
                
                detection_list.append({
                    'class': class_name,
                    'class_id': class_idx,
                    'confidence': round(confidence, 3),
                    'bbox': {
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    }
                })
        
        return detection_list, results[0].plot()  # plot() returns annotated image
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return [], image


def main():
    """Main loop"""
    logger.info("=" * 60)
    logger.info("Raspberry Pi YOLO Detection Client (Local Inference)")
    logger.info("=" * 60)
    
    # Initialize Firebase
    db = None
    if SEND_TO_FIREBASE:
        db = initialize_firebase()
        if db:
            # Set Raspberry Pi status in Firebase
            try:
                db.child("system").child("raspberry_pi_status").set(1)
                logger.info("Raspberry Pi status set in Firebase")
            except Exception as e:
                logger.warning(f"Could not set status in Firebase: {e}")
    
    # Load YOLO model
    model = load_yolo_model(MODEL_PATH)
    if not model:
        return
    
    # Initialize Raspberry Pi Camera
    picam2 = initialize_camera()
    if not picam2:
        logger.error("Camera initialization failed!")
        logger.error("Please check:")
        logger.error("1. Camera is properly connected")
        logger.error("2. Camera is enabled: sudo raspi-config")
        logger.error("3. Reboot after enabling camera")
        return
    
    logger.info(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {1/CAPTURE_INTERVAL:.1f} FPS")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"Firebase: {'Enabled' if db else 'Disabled'}")
    logger.info("Ready! Starting detection loop...\n")
    
    frame_count = 0
    total_detections = 0
    last_firebase_update = 0
    
    try:
        while True:
            # Capture image from Raspberry Pi Camera
            image = capture_image_picam2(picam2)
            
            if image is None:
                time.sleep(CAPTURE_INTERVAL)
                continue
            
            frame_count += 1
            
            # Run detection
            start_time = time.time()
            detections, annotated_image = run_detection(model, image)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Send to Firebase if detections found and enough time passed
            current_time = time.time()
            if detections and db and (current_time - last_firebase_update) >= FIREBASE_UPDATE_INTERVAL:
                send_detections_to_firebase(db, detections, frame_count, inference_time)
                last_firebase_update = current_time
            
            if detections:
                total_detections += len(detections)
                if frame_count % 10 == 0:
                    logger.info(
                        f"Frame {frame_count}: {len(detections)} objects detected "
                        f"({inference_time:.1f}ms)"
                    )
                    for det in detections[:3]:  # Show first 3 detections
                        logger.info(
                            f"   - {det['class']}: {det['confidence']:.2%} "
                            f"at ({det['bbox']['xmin']}, {det['bbox']['ymin']})"
                        )
            
            # Display annotated image
            if SHOW_DISPLAY:
                cv2.imshow('Raspberry Pi YOLO Detection', annotated_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quitting...")
                    break
            
            time.sleep(CAPTURE_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("\nStopping...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if picam2:
            picam2.stop()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
        
        # Update Firebase status on exit
        if db:
            try:
                db.child("system").child("raspberry_pi_status").set(0)
                logger.info("Updated Firebase status on exit")
            except:
                pass
        
        logger.info(f"Total frames: {frame_count}, Total detections: {total_detections}")

if __name__ == "__main__":
    main()

