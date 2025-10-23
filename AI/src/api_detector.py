import argparse
import base64
import json
from pathlib import Path
from typing import Dict, List, Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hosted API object detection client (cars, persons, emergency)"
    )
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument(
        "--endpoint",
        required=True,
        type=str,
        help="Inference API endpoint URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key or bearer token if required",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="person,car,truck,bus,motorcycle,bicycle",
        help="Comma-separated class names to keep",
    )
    return parser.parse_args()


def encode_image_to_base64(image_path: Path) -> str:
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def call_inference_api(endpoint: str, api_key: str | None, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return 2

    img_b64 = encode_image_to_base64(img_path)
    keep = [c.strip().lower() for c in args.classes.split(",") if c.strip()]

    # This payload is generic; match your provider's schema as needed.
    payload = {
        "image_base64": img_b64,
        "classes": keep,
        "return": ["boxes", "scores", "classes"],
    }

    try:
        data = call_inference_api(args.endpoint, args.api_key, payload)
    except requests.HTTPError as e:
        print(f"HTTP error: {e.response.status_code} {e.response.text}")
        return 3
    except Exception as e:
        print(f"Request failed: {e}")
        return 3

    # Expecting a response like: {"detections": [{"box": [x1,y1,x2,y2], "class": "car", "score": 0.92}, ...]}
    detections: List[Dict[str, Any]] = data.get("detections", [])
    if not detections:
        print("No detections")
        return 0

    # Print concise results
    for det in detections:
        cls_name = det.get("class", "?")
        score = det.get("score", 0.0)
        box = det.get("box", [])
        print(f"{cls_name:12s} {score:0.2f} box={box}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


