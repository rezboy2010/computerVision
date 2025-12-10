import mediapipe as mp
from rendering import *
from config import *
import cv2


bow_img = cv2.imread('static/images/arrow.png', cv2.IMREAD_UNCHANGED)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

with mp_pose.Pose(min_detection_confidence=0.8) as pose_detector, mp_hands.Hands() as hand_detector:
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, SCREEN_W, SCREEN_H)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_image = frame.copy()

        results = pose_detector.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[20]
            draw_bow(res_image, bow_img, left_wrist.x + BOW_SHIFT_X, left_wrist.y + BOW_SHIFT_Y)

        cv2.imshow(WIN_NAME, res_image)

        if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
