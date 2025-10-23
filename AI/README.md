# Road Image/Video Detector (Cars, People, Emergency Vehicles)

Minimal, efficient Python tool using Ultralytics YOLO to detect road users (person, car, truck, bus, motorcycle, bicycle) and flag potential emergency vehicles via a simple red/blue-lights heuristic.

Note: Use Python 3.11 for smooth installation. Newer versions (e.g., 3.14) may lack prebuilt wheels for NumPy/Torch on Windows and will try to build from source.

## Quick Start

1) Install Python 3.11 (Windows) from `https://www.python.org/downloads/windows/` and check "Add Python to PATH".

2) Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
```

3) Run detection on an image:

```bash
python src/detector.py --source image --input path/to/image.jpg --show --save
```

4) Run on a video file:

```bash
python src/detector.py --source video --input path/to/video.mp4 --save --output runs/annotated.mp4
```

5) Run with a webcam:

```bash
python src/detector.py --source webcam --show --save --output runs/annotated.mp4
```

## Options

- `--model`: YOLO weights, e.g. `yolov8n.pt` (fastest), `yolov8s.pt` (better), `yolov8m.pt`.
- `--conf`: confidence threshold (default 0.3).
- `--classes`: subset of classes to keep, comma-separated.
- `--emergency`: enable heuristic to flag potential emergency vehicles (checks strong red/blue pixels inside vehicle boxes).
- `--suppress-person-in-vehicle` / `--no-suppress-person-in-vehicle`: hide person boxes that are largely inside vehicle boxes (on by default). Helps avoid counting riders/passengers as separate persons.
- `--keep-motorcycle-riders`: when suppressing persons-in-vehicles, keep persons inside motorcycles.
- `--min-person-area N`: drop very small person boxes (< N pixels).
- `--output-dir`: output directory used when processing single images or folders.


## Batch Folder Mode

Process all images in a directory tree and save annotated results (mirrors the folder structure):

```bash
python src/detector.py --source folder --input assets/ --output-dir runs/ --classes person,car,truck,bus --emergency --min-person-area 600
```
- `--max-size`: inference size; reduce for speed (e.g., 640) or raise for accuracy.
- `--device`: `cuda` if you have an NVIDIA GPU; falls back to CPU.

## Notes on Efficiency

- Use `yolov8n.pt` for maximum speed; upgrade to `yolov8s.pt` if you need more accuracy.
- Lower `--max-size` (e.g., 640) for faster inference.
- Limit classes with `--classes` to reduce post-processing.
- Prefer GPU (`--device cuda`) if available.

## Emergency Vehicle Heuristic

This project includes a simple color-based heuristic that flags vehicle detections as "EMERGENCY?" when a notable fraction of pixels in the box are high-saturation red or blue. This is a lightweight indicator and may produce false positives/negatives. For production-grade performance, consider fine-tuning a dedicated emergency-vehicle class or using additional cues (siren pattern detection, text OCR for "AMBULANCE", "POLICE").

## Example Commands

```bash
# Image, save annotated output
python src/detector.py --source image --input assets/frame.jpg --save --output runs/frame_annotated.jpg --emergency

# Video, show + save
python src/detector.py --source video --input assets/drive.mp4 --show --save --output runs/drive_annotated.mp4 --model yolov8s.pt --conf 0.35 --max-size 768

# Webcam at index 1
python src/detector.py --source webcam --webcam-index 1 --show --emergency
```

## Alternative: Hosted API (no heavy local installs)

If installing PyTorch/numpy on Windows is difficult, use the lightweight API client.

1) Install minimal dependencies:

```bash
pip install -r requirements-api.txt
```

2) Call your provider's endpoint (replace URL and key):

```bash
python src/api_detector.py --image assets/frame.jpg --endpoint https://YOUR_PROVIDER/infer --api-key YOUR_KEY --classes person,car,truck,bus
```

The script prints each detection with class, score, and box coordinates.

## Assets

Put your test images/videos under an `assets/` folder (optional).


