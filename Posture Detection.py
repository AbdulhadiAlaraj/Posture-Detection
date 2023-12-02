import cv2
import mediapipe as mp
import numpy as np
import pickle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the logistic regression model
with open('Logistic_Regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Check if any landmarks are detected
    if results.pose_landmarks:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        pose_data = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()

        # Make sure the data shape matches what the model expects
            # Predict posture
        posture_class = model.predict(pose_data.reshape(1, -1))[0]
        posture_text = "Good Posture" if posture_class == 1 else "Adjust Posture"

            # Display the posture text
        cv2.putText(frame, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the landmarks
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the frame
    cv2.imshow('Sitting Posture Detection', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
