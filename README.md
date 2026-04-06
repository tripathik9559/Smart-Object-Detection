# 🎯 Smart Object Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red?style=for-the-badge&logo=opencv)
![Model](https://img.shields.io/badge/Model-SSD%20MobileNet%20V3-orange?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-COCO%2080%20Classes-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A real-time AI-powered object detection system with zone-based intrusion alerts, speed tracking, CSV logging, and a live HUD dashboard.**  
Works with webcam or any video file — runs fully offline, no cloud needed.

[✨ Features](#-features) • [🤖 How It Works](#-how-it-works) • [🚀 Getting Started](#-getting-started) • [⚙️ Configuration](#️-configuration)

</div>

---

## 🌟 Why This Project?

Most object detection demos just draw a box and stop there. This system goes further — it tracks objects across frames, estimates their speed, monitors a configurable restricted zone, logs everything to CSV, and announces what it sees using text-to-speech. Built to be actually useful, not just a demo.

> ✅ Works fully offline • ✅ 80 COCO object classes • ✅ Webcam or video file input

---

## ✨ Features

### 🎯 Real-Time Object Detection
- **SSD MobileNet V3** model — fast and accurate
- Detects **80 COCO classes** — person, car, bottle, laptop, phone, and more
- Confidence threshold filtering + **Non-Maximum Suppression (NMS)**
- Color-coded bounding boxes with corner markers per class

### 📊 Object Counting & HUD Dashboard
- Live per-class object count on screen
- Semi-transparent **HUD overlay** showing:
  - FPS (frames per second)
  - Session timer
  - Total zone alert count
  - Top 3 detected objects in current frame

### 🚨 Zone-Based Intrusion Alert
- Configurable **restricted zone** drawn on screen
- When a person/vehicle enters the zone:
  - Red flashing **"WARNING: ZONE INTRUSION DETECTED"** banner
  - Instant **voice alert** via text-to-speech
  - Alert logged with timestamp to CSV
- Zone can be toggled ON/OFF anytime with `Z` key

### 🏎️ Speed Estimation
- **Centroid-based tracker** assigns unique IDs to objects across frames
- Tracks displacement over time → estimates speed in **km/h**
- Speed displayed live on each object's label
- Configurable `PIXELS_PER_METER` calibration constant

### 📁 CSV Session Logging
- Every detected object is logged with:
  - Timestamp, frame number, class label
  - Confidence score, center position (cx, cy)
  - Estimated speed, zone alert flag
- Auto-saved every **30 seconds** + on program exit
- Stored in `logs/session_YYYYMMDD_HHMMSS.csv`

### 🔊 Voice Announcements
- Speaks a summary of what it sees every **6 seconds**
- Example: *"I see 2 person, 1 car"*
- Fully threaded — non-blocking, doesn't slow detection

---

## 🤖 How It Works

```
Webcam / Video File
        │
        ▼
  Frame Capture (OpenCV)
        │
        ▼
  SSD MobileNet V3 Detection
  (confThreshold=0.4, NMS=0.2)
        │
        ▼
  ┌─────┴──────────────────────────┐
  │                                │
  ▼                                ▼
Centroid Tracker              Zone Check
(assign IDs, track            (is object inside
 displacement)                 restricted area?)
  │                                │
  ▼                                ▼
Speed Estimation              Intrusion Alert
(km/h via px/s)               (banner + voice + CSV)
  │
  ▼
CSV Logger → logs/session_*.csv
  │
  ▼
HUD Overlay + Display Frame
```

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.9+ |
| **Computer Vision** | OpenCV 4.5+ (DNN module) |
| **Detection Model** | SSD MobileNet V3 Large (COCO 2020) |
| **Object Classes** | COCO Dataset — 80 classes |
| **Tracking** | Custom centroid-based tracker |
| **Voice** | pyttsx3 (offline TTS, threaded) |
| **Logging** | Python csv module |
| **Input** | Webcam (index 0) or any video file |

---

## 🎯 Detectable Objects (COCO 80 Classes)

Some examples of what this system can detect:

| Category | Objects |
|----------|---------|
| **People** | person |
| **Vehicles** | car, motorcycle, bicycle, bus, truck |
| **Electronics** | laptop, cell phone, keyboard, mouse, TV |
| **Kitchen** | bottle, cup, fork, knife, bowl |
| **Furniture** | chair, couch, bed, dining table |
| **Animals** | cat, dog, bird, horse, cow |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or above
- Webcam (or a video file)
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/tripathik9559/Smart-Object-Detection.git
cd Smart-Object-Detection

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# Linux only — for voice support:
sudo apt install espeak -y
```

### Run

```bash
# With webcam (default)
python main.py

# With a video file
python main.py --video path/to/your/video.mp4
```

### Check Available Cameras

```bash
python camera_test.py
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the program |
| `P` | Pause / Resume |
| `Z` | Toggle restricted zone ON / OFF |
| `S` | Force-save CSV log immediately |

---

## ⚙️ Configuration

All config is at the top of `Detector.py`:

```python
# ── Restricted Zone (fractions of frame: x1, y1, x2, y2) ──
ZONE_RECT = (0.55, 0.30, 0.95, 0.85)   # right-centre by default

# ── Which classes trigger zone alert ──
ZONE_CLASSES = {"person", "car", "motorcycle", "bicycle"}

# ── Speed estimation ──
PIXELS_PER_METER = 40      # calibrate based on your camera/scene

# ── Voice interval ──
speech_interval = 6        # seconds between voice announcements

# ── CSV auto-save interval ──
AUTO_SAVE_SECS = 30        # seconds between auto-saves
```

---

## 📁 Project Structure

```
Smart-Object-Detection/
├── main.py              # Entry point — supports --video flag
├── Detector.py          # Core logic — detection, tracking, alerts, HUD
├── sound.py             # Threaded TTS voice module
├── camera_test.py       # Utility to find available camera indexes
├── requirements.txt
├── model_data/
│   ├── frozen_inference_graph.pb              # SSD MobileNet V3 weights
│   ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt  # Model config
│   └── coco.names                             # 80 COCO class labels
├── extra/               # Experimental scripts (motion, human detection)
│   ├── motion.py
│   ├── human.py
│   └── human_motion.py
└── logs/                # Auto-created — CSV session logs saved here
```

---

## 📊 CSV Log Format

Each session creates `logs/session_YYYYMMDD_HHMMSS.csv`:

```csv
timestamp, frame, object,  confidence, cx,  cy,  speed_kmh, zone_alert
14:22:01,  45,    person,  0.87,       312, 240, 3.2,       YES
14:22:01,  45,    car,     0.91,       510, 300, 12.5,
14:22:07,  167,   laptop,  0.76,       200, 180, 0.0,
```

---

## 🛡️ Common Issues & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `Cannot open video source` | Check webcam connection or use `--video` flag |
| No voice on Linux | `sudo apt install espeak -y` |
| Black window appears | Run `python camera_test.py` to check camera index |
| `frozen_inference_graph.pb not found` | Re-extract the zip — model_data/ folder must be intact |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Kartikey Kumar Tripathi**  
🔗 [GitHub](https://github.com/tripathik9559)

---

<div align="center">

**⭐ If this project was useful, please give it a star! ⭐**

*Built with ❤️ for making computer vision practical and accessible.*

</div>
