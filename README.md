# Posture Detection w/ MediaPipe and OpenCV
## Introduction
The Posture Detection project is a machine learning-based system designed to detect and classify human posture in real-time using video input. Utilizing various algorithms and real-time processing, this project aims to assist users in maintaining proper posture, especially during prolonged periods of sitting.

## Features
- **Real-Time Posture Detection:** Leverages a webcam to monitor the user's posture continuously.
- **Pose Estimation:** Utilizes MediaPipe's Pose solution for accurate pose landmark detection.
- **Posture Classification:** Employs a machine learning model to classify the posture as 'Good' or 'Needs Adjustment'.
- **Feedback System:** Provides real-time visual feedback on the user's current posture.
## Components
- **ML_Pipeline.py:** Sets up the machine learning pipeline, including data preprocessing, model training (using various algorithms), and evaluation.
- **Posture Detection.py:** The main script for real-time posture monitoring, leveraging the webcam and ML model to provide immediate posture assessment.
- **captureData.py:** Similar to the main script but potentially used for data collection or additional testing purposes.
## Installation
To set up the project, follow these steps:
- Clone the repository.
- Install the required dependencies:

``` pip install -r requirements.txt ```

Run ```Posture Detection.py``` to start the posture detection in real-time.
## Usage
Ensure your webcam is enabled and properly positioned.
Execute ```Posture Detection.py``` to start monitoring.
The system will display your current posture status on the screen.
## Technologies Used
- Python
- OpenCV (cv2)
- MediaPipe
- Scikit-Learn
- NumPy
