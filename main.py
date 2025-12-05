import mediapipe as mp
import cv2


with mp.solutions.pose.Pose() as pose_detector:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        if cv2.waitKey(1) & 0xFF == 27:
            break

        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(img)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

        res_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Arrow of Fate", res_image)
