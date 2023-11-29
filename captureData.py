import csv
import os
import numpy as np
import cv2
import mediapipe as mp
# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
class_name = "Proper"
key = 0
cap = cv2.VideoCapture(0)
# Check if the camera opened successfully.
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
# Capture frames from the camera.
while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Ignoring empty camera frame.")
        continue
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)
    # Draw the pose landmarks on the frame.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    if not os.path.isfile('coords.csv'):
        try:
            # Ensure results.pose_landmarks exists before accessing coordinates
            if results.pose_landmarks:
                coordinates = len(results.pose_landmarks.landmark)
                print(coordinates)
                landmarks = ['class']
                for val in range(1, coordinates+1):
                    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
                with open('coords.csv', mode='w', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(landmarks)
                print("coords.csv has been created.")
                key=1
                cap.release()
                cv2.destroyAllWindows()
        except AttributeError:
            print("No landmarks to write to coords.csv.")
    if results.pose_landmarks and key==0:
        poses = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in poses]).flatten())
        pose_row.insert(0, class_name)
        with open('coords.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(pose_row)
    cv2.imshow('Live Pose Tracking', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
pose.close()
cv2.destroyAllWindows()
#testing venv ignore