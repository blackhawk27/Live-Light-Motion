

import cv2
import mediapipe as mp
from osc_bridge import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# For webcam input:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")

            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        ret, frame = cap.read()

        # Get landmarks koordinates
        try:
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            landmarks = results.pose_landmarks.landmark
            osc_msg = [landmarks[mp_pose.PoseLandmark.NOSE].x * frame_width,
                       landmarks[mp_pose.PoseLandmark.NOSE].y * frame_height,
                       landmarks[mp_pose.PoseLandmark.NOSE].z * frame_height]
            # print(osc_msg)

            send_osc_msg(osc_msg)

        except:
            pass

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

        #print(cv2.getWindowImageRect('MediaPipe Pose'))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
