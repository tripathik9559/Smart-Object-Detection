"""
main.py — Smart Object Detection System
Author  : Kartikey Kumar Tripathi

Usage:
  python main.py                   -> webcam (default)
  python main.py --video path.mp4  -> video file

Keyboard controls while running:
  Q -> Quit
  P -> Pause / Resume
  Z -> Toggle restricted zone ON/OFF
  S -> Save CSV log immediately
"""

import argparse
import os
from Detector import Detector


def main():
    parser = argparse.ArgumentParser(description="Smart Object Detection System")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. Leave empty to use webcam.")
    args = parser.parse_args()

    videoPath   = args.video if args.video else 0
    configPath  = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath   = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    print("=" * 55)
    print("   Smart Object Detection System")
    print("   Author : Kartikey Kumar Tripathi")
    print("=" * 55)
    print(f"   Source : {'Webcam' if videoPath == 0 else videoPath}")
    print("=" * 55 + "\n")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()


if __name__ == "__main__":
    main()
