"""
Flask server for toy car detection using YOLO.
Receives images from ESP32-CAM and returns JSON detections.
"""
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os
from pathlib import Path

app = Flask(__name__)

# Load model - will be set after training
MODEL_PATH = "runs/detect/train/weights/best.pt"
model = None

def load_model():
    """Load the trained YOLO model."""
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Using default YOLOv8n model for testing")
            model = YOLO("yolov8n.pt")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}")
        print("⚠️  Using default YOLOv8n model for testing")
        print("💡 Train your model first, then update MODEL_PATH if needed")
        model = YOLO("yolov8n.pt")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/detect', methods=['POST'])
def detect():
    """
    Receive image from ESP32-CAM and return detections.
    Accepts: 
    - multipart/form-data with 'image' field (preferred)
    - raw binary JPEG data with Content-Type: image/jpeg
    Returns: JSON array of detections
    """
    try:
        img_bytes = None
        
        # Try multipart/form-data first
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "Empty file"}), 400
            img_bytes = file.read()
        # Try raw binary data
        elif request.content_type and 'image' in request.content_type:
            img_bytes = request.data
        # Try request data as fallback
        elif request.data:
            img_bytes = request.data
        
        if not img_bytes or len(img_bytes) == 0:
            return jsonify({"error": "No image data provided. Send as multipart/form-data with 'image' field or raw binary JPEG."}), 400
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Run inference
        results = model(img, conf=0.25, imgsz=320, verbose=False)[0]
        
        # Extract detections
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2)
                })
        
        return jsonify({
            "detections": detections,
            "count": len(detections)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Toy Car Detection Server...")
    print("=" * 50)
    load_model()
    print("=" * 50)
    print("📡 Server listening on http://0.0.0.0:5000")
    print("💡 ESP32-CAM should POST images to http://YOUR_IP:5000/detect")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)


