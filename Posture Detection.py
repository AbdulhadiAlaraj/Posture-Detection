import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Coordinates for shoulders
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        head = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[mp_pose.PoseLandmark.NOSE.value].y]

        # Calculate approximate neck position
        neck = [(left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2]

    # Coordinates for hips
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        # Inferring mid-spine position as midpoint between hips
        mid_spine = [(left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2]


        # Calculate angles
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, head)
        neck_angle = calculate_angle(head, neck, mid_spine)

        # Sitting posture analysis
        posture_text = "Adjust Posture"
        if shoulder_angle > 45:  # Example thresholds
            posture_text = "Good Posture"

        # Display angles and posture text
        cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Neck Angle: {neck_angle:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, posture_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Sitting Posture Detection', frame)

    if cv2.waitKey(5) & 0xFF == ord('qqqqq'):
        break

cap.release()
cv2.destroyAllWindows()

