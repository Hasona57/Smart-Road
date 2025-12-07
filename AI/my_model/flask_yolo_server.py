"""
Flask YOLO Detection Server
Receives images from ESP32-CAM, runs YOLO detection, and returns JSON results

Usage: python flask_yolo_server.py --model my_model.pt --port 5000
"""

import os
import sys
import argparse
import time
import threading
import base64
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from ESP32

# Global variables
model = None
labels = None
detection_count = 0
latest_image = None
latest_image_lock = threading.Lock()
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
show_window = False
executor = ThreadPoolExecutor(max_workers=2)  # Process images in parallel

def parse_arguments():
    parser = argparse.ArgumentParser(description='Flask YOLO Detection Server')
    parser.add_argument('--model', help='Path to YOLO model file', required=True)
    parser.add_argument('--port', help='Server port (default: 5000)', type=int, default=5000)
    parser.add_argument('--host', help='Server host (default: 0.0.0.0)', default='0.0.0.0')
    parser.add_argument('--thresh', help='Confidence threshold (default: 0.5)', type=float, default=0.5)
    parser.add_argument('--display', help='Show OpenCV window with detections', action='store_true')
    parser.add_argument('--imgsz', help='Inference size (default: 640 for accuracy, use 320 for speed)', type=int, default=640)
    parser.add_argument('--device', help='Device: cpu or cuda (default: cpu)', default='cpu')
    return parser.parse_args()

def load_model(model_path):
    """Load YOLO model"""
    global model, labels
    if not os.path.exists(model_path):
        print(f'ERROR: Model file not found: {model_path}')
        sys.exit(1)
    
    print(f'Loading YOLO model from {model_path}...')
    model = YOLO(model_path, task='detect')
    labels = model.names
    
    # Get model info
    from ultralytics.utils import __version__ as ultralytics_version
    print(f'✅ Model loaded successfully!')
    print(f'   Ultralytics version: {ultralytics_version}')
    print(f'   Model type: {type(model.model).__name__}')
    print(f'   Classes: {len(labels)}')
    print(f'   Class names: {list(labels.values())[:10]}{"..." if len(labels) > 10 else ""}')

def process_detection(image_bytes):
    """Process detection in background thread"""
    global detection_count, latest_image
    
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None
        
        # Run YOLO detection - optimized settings with latest YOLO API
        # Use latest inference parameters for best performance
        inference_params = {
            'verbose': False,
            'imgsz': app.config.get('IMGSZ', 640),
            'conf': app.config['THRESHOLD'],
            'device': app.config.get('DEVICE', 'cpu'),
            'half': False,  # Set to True if you have GPU with FP16 support
            'agnostic_nms': False,  # Class-agnostic NMS
            'max_det': 300,  # Maximum detections per image
        }
        
        results = model(image, **inference_params)
        
        detections = results[0].boxes
        
        # Process detections
        detection_list = []
        object_count = 0
        
        # Pre-allocate for speed
        if len(detections) > 0:
            annotated_image = image.copy()
            
            for i in range(len(detections)):
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int).tolist()
                
                classidx = int(detections[i].cls.item())
                classname = labels[classidx]
                conf = float(detections[i].conf.item())
                
                if conf > app.config['THRESHOLD']:
                    detection_list.append({
                        'class': classname,
                        'class_id': classidx,
                        'confidence': round(conf, 3),
                        'bbox': {
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax
                        }
                    })
                    
                    # Draw bounding box (only if needed for display)
                    if show_window or True:
                        color = bbox_colors[classidx % len(bbox_colors)]
                        cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), color, 2)
                        
                        # Draw label
                        label = f'{classname}: {int(conf*100)}%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_ymin = max(ymin, labelSize[1] + 10)
                        cv2.rectangle(annotated_image, (xmin, label_ymin-labelSize[1]-10), 
                                    (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                        cv2.putText(annotated_image, label, (xmin, label_ymin-7), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    object_count += 1
            
            # Draw object count
            if show_window or True:
                cv2.putText(annotated_image, f'Objects: {object_count}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Store latest image
            with latest_image_lock:
                latest_image = annotated_image.copy()
        else:
            # No detections - still store image
            with latest_image_lock:
                latest_image = image.copy()
        
        detection_count += 1
        
        return {
            'success': True,
            'detections': detection_list,
            'count': len(detection_list),
            'image_size': {
                'width': int(image.shape[1]),
                'height': int(image.shape[0])
            },
            'total_processed': detection_count
        }
        
    except Exception as e:
        print(f'Error processing image: {str(e)}')
        return {'error': str(e), 'success': False}

@app.route('/detect', methods=['POST'])
def detect():
    """Receive image from ESP32-CAM and run YOLO detection - optimized for speed and reliability"""
    try:
        # Get image data from request
        if 'image' in request.files:
            image_file = request.files['image']
            image_bytes = image_file.read()
        else:
            image_bytes = request.data
        
        if len(image_bytes) == 0:
            return jsonify({'error': 'No image data received'}), 400
        
        # Process detection (synchronous for now, but optimized)
        result = process_detection(image_bytes)
        
        if result is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Display in OpenCV window if requested
        if show_window and latest_image is not None:
            cv2.imshow('ESP32-CAM YOLO Detection', latest_image)
            cv2.waitKey(1)
        
        # Return response with keep-alive headers for better connection handling
        response = jsonify(result)
        response.headers['Connection'] = 'close'  # Explicitly close connection
        response.headers['Content-Type'] = 'application/json'
        return response, 200
        
    except Exception as e:
        print(f'Error in detect endpoint: {str(e)}')
        error_response = jsonify({'error': str(e), 'success': False})
        error_response.headers['Connection'] = 'close'
        return error_response, 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'total_detections': detection_count
    }), 200

@app.route('/view', methods=['GET'])
def view():
    """View latest detection image"""
    with latest_image_lock:
        if latest_image is None:
            return jsonify({'error': 'No image available yet'}), 404
        
        # Encode image as JPEG
        _, buffer = cv2.imencode('.jpg', latest_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        return Response(frame_bytes, mimetype='image/jpeg')

@app.route('/stream', methods=['GET'])
def stream():
    """MJPEG stream of detection results"""
    def generate():
        while True:
            with latest_image_lock:
                if latest_image is not None:
                    # Encode image as JPEG
                    _, buffer = cv2.imencode('.jpg', latest_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET'])
def index():
    """Server info page with live view"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Detection Server</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f0f0f0;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
            }}
            .status {{
                background-color: #e8f5e9;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }}
            .video-container {{
                text-align: center;
                margin: 20px 0;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 2px solid #333;
                border-radius: 5px;
            }}
            .info {{
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }}
        </style>
        <script>
            // Auto-refresh image every 500ms
            setInterval(function() {{
                var img = document.getElementById('detectionImage');
                img.src = '/view?t=' + new Date().getTime();
            }}, 500);
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🎯 YOLO Detection Server</h1>
            
            <div class="status">
                <p><strong>Status:</strong> Running ✅</p>
                <p><strong>Model:</strong> {'Loaded' if model is not None else 'Not loaded'}</p>
                <p><strong>Total detections processed:</strong> {detection_count}</p>
            </div>
            
            <div class="video-container">
                <h2>Live Detection View</h2>
                <img id="detectionImage" src="/view" alt="Detection View" 
                     onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'640\\' height=\\'480\\'%3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\'%3EWaiting for images...%3C/text%3E%3C/svg%3E';">
                <p><em>Image updates automatically every 500ms</em></p>
            </div>
            
            <div class="info">
                <h2>📡 Endpoints:</h2>
                <ul>
                    <li><b>POST /detect</b> - Send image for detection (returns JSON)</li>
                    <li><b>GET /view</b> - View latest detection image</li>
                    <li><b>GET /stream</b> - MJPEG stream of detections</li>
                    <li><b>GET /health</b> - Health check</li>
                </ul>
            </div>
            
            <div class="info">
                <h2>🔗 Alternative Views:</h2>
                <ul>
                    <li><a href="/view" target="_blank">View Latest Image</a></li>
                    <li><a href="/stream" target="_blank">MJPEG Stream</a></li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """, 200

if __name__ == '__main__':
    args = parse_arguments()
    
    # Load model
    load_model(args.model)
    
    # Configure Flask app
    app.config['THRESHOLD'] = args.thresh
    app.config['IMGSZ'] = args.imgsz
    app.config['DEVICE'] = args.device
    show_window = args.display
    
    print(f'\n📡 ESP32-CAM Detection Server')
    print(f'   Waiting for images from ESP32-CAM via POST /detect')
    
    # Start server
    print(f'\n🚀 Starting Flask server...')
    print(f'   Host: {args.host}')
    print(f'   Port: {args.port}')
    print(f'   Threshold: {args.thresh}')
    print(f'   OpenCV Display: {"Enabled" if show_window else "Disabled"}')
    print(f'\n📡 Server URLs:')
    print(f'   - http://localhost:{args.port} (Web interface)')
    print(f'   - http://127.0.0.1:{args.port}')
    
    # Get local IP address
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f'   - http://{local_ip}:{args.port}')
        print(f'\n💡 Use this IP in your ESP32 code: {local_ip}')
    except:
        pass
    
    print(f'\n🌐 View detections in browser: http://localhost:{args.port}')
    
    if show_window:
        print(f'\n🖼️  OpenCV window will open when first frame is received')
        print(f'   Press \'q\' in the window to quit')
    
    print(f'\n✅ Server ready! Waiting for ESP32-CAM requests...\n')
    
    try:
        # Run with optimized settings for reliability
        app.run(
            host=args.host, 
            port=args.port, 
            debug=False, 
            threaded=True
        )
    finally:
        if show_window:
            cv2.destroyAllWindows()

