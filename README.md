# Posture-Detection
Posture Detection with Mediapipe and OpenCV
Overview
This project aims to develop a real-time posture detection system using computer vision techniques. It utilizes a front-facing camera and the MediaPipe framework to detect and analyze the user's posture, particularly focusing on the sitting posture. The system provides visual feedback to encourage better ergonomic practices, especially for individuals who spend prolonged periods at a desk.

## Features:

**Real-Time Posture Detection:** Utilizes a camera to capture video feed and analyze posture in real-time.

**Pose Estimation:** Leverages MediaPipe's pose estimation capabilities to identify key body landmarks.

**Posture Analysis:** Analyzes posture using Machine Learning models. 

**Feedback System:** Provides on-screen feedback about the user’s posture, suggesting adjustments when necessary.

## Requirements
- Python 3.x
- OpenCV-Python
- MediaPipe
- NumPy

## Installation:
```pip install opencv-python mediapipe numpy```

## Usage
To start the posture detection, simply run the posture_detection.py script:
python posture_detection.py
Ensure your environment has good lighting and the camera is positioned to capture your upper body from the front.
