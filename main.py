import mediapipe as mp
from config import *
import numpy as np
import cv2


with (mp.solutions.pose.Pose() as pose_detector, mp.solutions.hands.Hands() as hand_detector):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.cvtColor(np.fliplr(frame), cv2.COLOR_BGR2RGB)
        results = pose_detector.process(img)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

        res_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(WIN_NAME, res_image)

        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()

cv2.destroyAllWindows()
