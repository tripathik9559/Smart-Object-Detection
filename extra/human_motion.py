import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize YOLOv8 Model
model = YOLO("yolov8n.pt")

# Video Capture
cap = cv2.VideoCapture("test_videos/street1.mp4")  # "test_videos/street1.mp4"

# Motion Tracking History
motion_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Detection
    results = model(frame)

    for result in results:
        boxes = result.boxes.data
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0:  # Class 0 is 'person'
                # Draw Bounding Box
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )

                # Crop Person Region
                person_crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
                if person_crop.size == 0:
                    continue

                # Convert Crop to RGB for MediaPipe
                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(person_rgb)

                # Draw Pose Landmarks on the Cropped Region
                if results_pose.pose_landmarks:
                    mp_draw.draw_landmarks(
                        person_crop,
                        results_pose.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(
                            color=(0, 0, 255), thickness=2, circle_radius=2
                        ),
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                    )

                # Place the cropped region back on the main frame
                frame[int(y1) : int(y2), int(x1) : int(x2)] = person_crop

                # Track Movement
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if cls not in motion_history:
                    motion_history[cls] = center
                else:
                    prev_center = motion_history[cls]
                    speed = np.linalg.norm(np.array(center) - np.array(prev_center))
                    motion_history[cls] = center

                    if speed < 5:
                        movement = "Standing"
                    elif speed < 20:
                        movement = "Walking"
                    else:
                        movement = "Running"

                    # Green Block for Label
                    label_size = cv2.getTextSize(
                        movement, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )[0]
                    label_x1 = int(x1)
                    label_y1 = int(y1) - 30
                    label_x2 = int(x1) + label_size[0] + 10
                    label_y2 = int(y1) - 5

                    cv2.rectangle(
                        frame,
                        (label_x1, label_y1),
                        (label_x2, label_y2),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        movement,
                        (label_x1 + 5, label_y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2,
                    )

    # Display the Result
    cv2.imshow("Multi-Person Motion & Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
