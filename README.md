# Smart Object Detection System

**Author:** Kartikey Kumar Tripathi  
**Tech Stack:** Python · OpenCV · SSD MobileNet V3 · COCO Dataset · pyttsx3

---

## What This Project Does

A real-time object detection system built with OpenCV's DNN module and SSD MobileNet V3 (trained on COCO — 80 object classes). Beyond basic detection, this system includes several practical features that make it useful for surveillance, traffic monitoring, and security applications.

---

## Features

| Feature | Details |
|---|---|
| **Real-time Detection** | SSD MobileNet V3, 80 COCO classes, 320×320 input |
| **Object Counting** | Per-class count displayed live on screen |
| **CSV Session Log** | Every detected object logged with timestamp, confidence, position, speed. Auto-saved every 30 seconds and on exit. Stored in `logs/` |
| **Speed Estimation** | Centroid-based tracker estimates object speed in km/h using frame-to-frame displacement |
| **Zone-based Alert** | Configurable restricted zone — triggers a red banner + voice alert when a person/vehicle enters |
| **HUD Dashboard** | Semi-transparent overlay showing FPS, session timer, alert count, top detected classes |
| **Voice Announcements** | Speaks what it sees every 6 seconds using pyttsx3 (threaded, non-blocking) |

---

## Project Structure

```
Smart-Object-Detection/
├── main.py              # Entry point (supports --video flag)
├── Detector.py          # Core detection + all feature logic
├── sound.py             # Threaded TTS voice module
├── camera_test.py       # Utility to check available cameras
├── requirements.txt
├── model_data/
│   ├── frozen_inference_graph.pb          # SSD MobileNet V3 weights
│   ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt  # Model config
│   └── coco.names                         # 80 class labels
└── logs/                # Auto-created — CSV session logs saved here
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# On Linux, also install espeak for voice:
sudo apt install espeak

# 2. Run with webcam
python main.py

# 3. Run with a video file
python main.py --video path/to/video.mp4
```

---

## Keyboard Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `P` | Pause / Resume |
| `Z` | Toggle restricted zone ON/OFF |
| `S` | Force-save CSV log immediately |

---

## Configuring the Restricted Zone

In `Detector.py`, edit these two lines at the top:

```python
# Zone position: (x1, y1, x2, y2) as fractions of frame size
ZONE_RECT = (0.55, 0.30, 0.95, 0.85)   # right-centre by default

# Which object classes trigger the zone alert
ZONE_CLASSES = {"person", "car", "motorcycle", "bicycle"}
```

---

## CSV Log Format

Each session generates a file in `logs/session_YYYYMMDD_HHMMSS.csv`:

```
timestamp, frame, object, confidence, cx, cy, speed_kmh, zone_alert
14:22:01,  45,    person, 0.87,       312, 240, 3.2,     YES
14:22:01,  45,    car,    0.91,       510, 300, 12.5,
```

---

## Resume Line

> **Smart Object Detection System** | Python, OpenCV, SSD MobileNet V3  
> Built a real-time object detection pipeline with zone-based intrusion alerts, centroid tracking for speed estimation, per-session CSV logging, and a live HUD dashboard. Detects 80 object classes from the COCO dataset via webcam or video input.
