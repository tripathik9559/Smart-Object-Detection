"""
Detector.py  —  Enhanced Real-Time Object Detection
Author : Kartikey Kumar Tripathi
Features added on top of base detection:
  - Object counting per frame (per class)
  - CSV session log  (auto-saved every 30 s + on exit, stored in logs/)
  - Zone-based intrusion alert  (configurable red zone, flashing banner)
  - Centroid-based speed estimation  (px/s -> km/h approximation)
  - On-screen HUD dashboard  (object counts, fps, session time, alert count)
  - Voice announcements via pyttsx3  (threaded, non-blocking)
Keyboard controls:
  Q = Quit       P = Pause/Resume
  Z = Toggle restricted zone    S = Save CSV immediately
"""

import cv2
import numpy as np
import time
import threading
import csv
import os
from datetime import datetime
from collections import defaultdict, deque
from sound import speak

np.random.seed(20)

# ── Zone config  (fractions of frame width/height, top-left → bottom-right) ──
ZONE_RECT    = (0.55, 0.30, 0.95, 0.85)        # right-centre region by default
ZONE_CLASSES = {"person", "car", "motorcycle", "bicycle"}  # classes that trigger alert

# ── Speed estimation ──────────────────────────────────────────────────────────
PIXELS_PER_METER = 40      # calibrate per camera / scene depth
SPEED_HISTORY    = 6       # number of frames to average over

# ── CSV logging ───────────────────────────────────────────────────────────────
LOG_DIR        = "logs"
AUTO_SAVE_SECS = 30        # flush buffer to disk every N seconds


class Detector:
    def __init__(self, videoPath, configPath, modelPath, classPath):
        self.videoPath  = videoPath
        self.configPath = configPath
        self.modelPath  = modelPath
        self.classPath  = classPath

        # ── Load SSD model ────────────────────────────────────────────────────
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

        # ── Voice ─────────────────────────────────────────────────────────────
        self.last_speech_time = time.time()
        self.speech_interval  = 6
        self.speech_lock      = threading.Lock()

        # ── CSV log ───────────────────────────────────────────────────────────
        os.makedirs(LOG_DIR, exist_ok=True)
        ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(LOG_DIR, f"session_{ts}.csv")
        self.log_rows = []
        self.last_save = time.time()
        self._init_csv()

        # ── Zone ──────────────────────────────────────────────────────────────
        self.zone_enabled    = True
        self.zone_alert_time = 0

        # ── Speed / centroid tracking ─────────────────────────────────────────
        self.track_history: dict = defaultdict(lambda: deque(maxlen=SPEED_HISTORY))
        self.next_id         = 0
        self.prev_centroids  = []   # list of (cx, cy, track_id)

        # ── Session stats ─────────────────────────────────────────────────────
        self.session_start = time.time()
        self.total_alerts  = 0
        self.frame_count   = 0

    # ─────────────────────────────────────────────────────────────────────────
    def readClasses(self):
        with open(self.classPath, 'r') as f:
            self.classesList = f.read().splitlines()
        self.classesList.insert(0, '__Background__')
        self.colorList = np.random.uniform(low=0, high=255,
                                           size=(len(self.classesList), 3))

    # ─── CSV helpers ──────────────────────────────────────────────────────────
    def _init_csv(self):
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ["timestamp", "frame", "object", "confidence",
                 "cx", "cy", "speed_kmh", "zone_alert"])

    def _flush_csv(self):
        if not self.log_rows:
            return
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerows(self.log_rows)
        self.log_rows.clear()
        print(f"[LOG] CSV saved → {self.log_path}")

    # ─── Utilities ────────────────────────────────────────────────────────────
    def ContrastTextColor(self, boxColor):
        lum = 0.299*boxColor[2] + 0.587*boxColor[1] + 0.114*boxColor[0]
        return (0, 0, 0) if lum > 127 else (255, 255, 255)

    def _get_zone_px(self, W, H):
        return (int(ZONE_RECT[0]*W), int(ZONE_RECT[1]*H),
                int(ZONE_RECT[2]*W), int(ZONE_RECT[3]*H))

    def _in_zone(self, cx, cy, W, H):
        x1, y1, x2, y2 = self._get_zone_px(W, H)
        return x1 <= cx <= x2 and y1 <= cy <= y2

    # ─── Centroid tracker ─────────────────────────────────────────────────────
    def _match_centroid(self, cx, cy, max_dist=80):
        best_id, best_d = None, max_dist
        for (px, py, pid) in self.prev_centroids:
            d = np.hypot(cx - px, cy - py)
            if d < best_d:
                best_d, best_id = d, pid
        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
        return best_id

    def _estimate_speed(self, track_id):
        hist = self.track_history[track_id]
        if len(hist) < 2:
            return 0.0
        t0, x0, y0 = hist[0]
        t1, x1, y1 = hist[-1]
        dt = t1 - t0
        if dt < 1e-3:
            return 0.0
        dist_m    = np.hypot(x1-x0, y1-y0) / PIXELS_PER_METER
        return round(dist_m / dt * 3.6, 1)   # km/h

    # ─── HUD dashboard ────────────────────────────────────────────────────────
    def _draw_hud(self, image, fps, object_count, session_secs):
        H, W   = image.shape[:2]
        pw, ph = 270, 155
        overlay = image.copy()
        cv2.rectangle(overlay, (8, H-ph-8), (pw+8, H-8), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        y = H - ph + 14
        def put(txt, col=(210,210,210), sc=0.52, th=1):
            nonlocal y
            cv2.putText(image, txt, (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
            y += 22

        put(f"FPS     : {int(fps)}", (80,255,80), 0.54, 1)
        put(f"Session : {int(session_secs//60):02d}:{int(session_secs%60):02d}")
        put(f"Alerts  : {self.total_alerts}",
            (80,80,255) if self.total_alerts else (210,210,210))
        zone_col = (0,220,80) if self.zone_enabled else (120,120,120)
        put(f"Zone    : {'ON [Z to off]' if self.zone_enabled else 'OFF [Z to on]'}", zone_col)
        top3 = sorted(object_count.items(), key=lambda x: -x[1])[:3]
        for lbl, cnt in top3:
            put(f"  {lbl}: {cnt}", (255,200,60), 0.47)

    # ─── Zone overlay ─────────────────────────────────────────────────────────
    def _draw_zone(self, image):
        H, W = image.shape[:2]
        x1, y1, x2, y2 = self._get_zone_px(W, H)
        ov = image.copy()
        cv2.rectangle(ov, (x1,y1), (x2,y2), (0,0,180), -1)
        cv2.addWeighted(ov, 0.18, image, 0.82, 0, image)
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(image, "RESTRICTED ZONE", (x1+6, y1+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,255), 2, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
        if not cap.isOpened():
            print("[ERROR] Cannot open video source.")
            return

        success, image = cap.read()
        startTime = time.time()
        paused    = False

        print("\n[INFO] Controls → Q: Quit | P: Pause | Z: Toggle Zone | S: Save CSV")
        print(f"[INFO] Session log : {self.log_path}\n")

        while success:
            if not paused:
                self.frame_count += 1
                now       = time.time()
                fps       = 1.0 / max(now - startTime, 1e-6)
                startTime = now
                H, W      = image.shape[:2]

                # ── Detection ─────────────────────────────────────────────────
                classLabelIDs, confidences, bboxs = self.net.detect(
                    image, confThreshold=0.4)

                bboxs       = list(bboxs)
                confidences = list(np.array(confidences).reshape(1,-1)[0])
                confidences = list(map(float, confidences))

                bboxIdx = cv2.dnn.NMSBoxes(
                    bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

                object_count  = {}
                zone_active   = False
                new_centroids = []

                if self.zone_enabled:
                    self._draw_zone(image)

                if len(bboxIdx) != 0:
                    for i in range(len(bboxIdx)):
                        idx   = np.squeeze(bboxIdx[i])
                        bbox  = bboxs[idx]
                        conf  = confidences[idx]
                        lid   = np.squeeze(classLabelIDs[idx])
                        lbl   = self.classesList[lid]
                        col   = [int(c) for c in self.colorList[lid]]
                        tcol  = self.ContrastTextColor(col)

                        x, y, w, h = bbox
                        cx, cy = x + w//2, y + h//2

                        # ── Speed tracking ────────────────────────────────────
                        tid = self._match_centroid(cx, cy)
                        new_centroids.append((cx, cy, tid))
                        self.track_history[tid].append((now, cx, cy))
                        spd = self._estimate_speed(tid)

                        # ── Zone check ────────────────────────────────────────
                        in_z = (self.zone_enabled and lbl in ZONE_CLASSES
                                and self._in_zone(cx, cy, W, H))
                        if in_z:
                            zone_active = True

                        # ── Draw bbox with corner lines ───────────────────────
                        bcol = (0, 0, 255) if in_z else col
                        cv2.rectangle(image, (x,y), (x+w,y+h), bcol,
                                      2 if in_z else 1)
                        lw = min(int(w*0.3), int(h*0.3))
                        for (px,py,dx,dy) in [
                            (x,   y,    1,  1), (x+w, y,   -1,  1),
                            (x,   y+h,  1, -1), (x+w, y+h, -1, -1)]:
                            cv2.line(image,(px,py),(px+dx*lw,py),bcol,4)
                            cv2.line(image,(px,py),(px,py+dy*lw),bcol,4)

                        # Label
                        txt = f"{lbl}:{conf:.2f}"
                        if spd > 0.5:
                            txt += f"  {spd}km/h"
                        cv2.putText(image, txt, (x, y-8),
                                    cv2.FONT_HERSHEY_PLAIN, 1, tcol, 2)

                        # Count
                        object_count[lbl] = object_count.get(lbl, 0) + 1

                        # CSV row
                        self.log_rows.append([
                            datetime.now().strftime("%H:%M:%S"),
                            self.frame_count, lbl, round(conf,3),
                            cx, cy, spd, "YES" if in_z else ""])

                self.prev_centroids = new_centroids

                # ── Zone intrusion banner ──────────────────────────────────────
                if zone_active and now - self.zone_alert_time > 5:
                    self.total_alerts   += 1
                    self.zone_alert_time = now
                    cv2.rectangle(image, (0,0), (W,40), (0,0,200), -1)
                    cv2.putText(image, "  WARNING: ZONE INTRUSION DETECTED",
                                (8,28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.78, (255,255,255), 2, cv2.LINE_AA)
                    threading.Thread(
                        target=speak,
                        args=("Warning! Object in restricted zone.",),
                        daemon=True).start()

                # ── Voice summary ──────────────────────────────────────────────
                if object_count:
                    summary = "I see " + ", ".join(
                        f"{c} {l}" for l, c in object_count.items())
                    with self.speech_lock:
                        if now - self.last_speech_time >= self.speech_interval:
                            threading.Thread(target=speak, args=(summary,),
                                             daemon=True).start()
                            self.last_speech_time = now

                # ── HUD ───────────────────────────────────────────────────────
                self._draw_hud(image, fps, object_count, now - self.session_start)

                # ── Auto CSV flush ─────────────────────────────────────────────
                if now - self.last_save >= AUTO_SAVE_SECS:
                    self._flush_csv()
                    self.last_save = now

                cv2.imshow("Smart Object Detector  |  Kartikey Kumar Tripathi", image)

            # ── Keyboard ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord('z'):
                self.zone_enabled = not self.zone_enabled
                print(f"[INFO] Zone {'enabled' if self.zone_enabled else 'disabled'}")
            elif key == ord('s'):
                self._flush_csv()

            if not paused:
                success, image = cap.read()

        # ── Cleanup ───────────────────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()
        self._flush_csv()
        print(f"\n[DONE] Frames: {self.frame_count} | Zone Alerts: {self.total_alerts}")
        print(f"[DONE] Log: {self.log_path}")
