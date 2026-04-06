import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0) #'test_videos\street3.mp4'

motion_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)

    if results_pose.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
        )

        landmarks = results_pose.pose_landmarks.landmark
        center_x = int(landmarks[mp_pose.PoseLandmark.NOSE].x * frame.shape[1])
        center_y = int(landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0])
        
        # Calculate a position above the head using the nose landmark
        hair_y = center_y - 100  # Move upwards to the hair position

        center = (center_x, center_y)

        if 'person' not in motion_history:
            motion_history['person'] = center
        else:
            prev_center = motion_history['person']
            speed = np.linalg.norm(np.array(center) - np.array(prev_center))
            motion_history['person'] = center

            if speed < 5:
                movement = "Standing"
            elif speed < 20:
                movement = "Walking"
            else:
                movement = "Running"

            # Display Movement Label above the person's head (on hair)
            label_size = cv2.getTextSize(movement, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            label_x1 = center_x - label_size[0] // 2
            label_y1 = hair_y - 20  # Adjust label placement above the head
            label_x2 = center_x + label_size[0] // 2
            label_y2 = hair_y + label_size[1]

            # Draw a background for the label
            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)
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
    cv2.imshow("Pose Detection & Movement Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
