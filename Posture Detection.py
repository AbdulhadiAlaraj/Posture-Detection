import cv2
import mediapipe as mp
import numpy as np
import pickle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

with open('Logistic_Regression.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark
        pose_data = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()

        posture_class = model.predict(pose_data.reshape(1, -1))[0]
        posture_text = "Good Posture" if posture_class == 1 else "Adjust Posture"

        cv2.putText(frame, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Sitting Posture Detection', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
