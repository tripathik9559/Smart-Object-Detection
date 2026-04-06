import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('test_videos/street1.mp4')

motion_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes.data
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

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
                    
                    cv2.putText(frame, f"{movement}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow('Human Motion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
