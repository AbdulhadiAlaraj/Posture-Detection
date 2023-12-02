import cv2
import mediapipe as mp
import numpy as np
import pickle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
with open('Logistic_Regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Check if landmarks are detected
    if results.pose_landmarks:
        # Extract and flatten landmarks
        landmarks = results.pose_landmarks.landmark
        pose_data = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten()

        # Predict posture
        posture_class = model.predict([pose_data])[0]
        posture_text = "Good Posture" if posture_class == 1 else "Adjust Posture"

        # Display the posture text
        cv2.putText(frame, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display frame
    cv2.imshow('Live Posture Detection', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
