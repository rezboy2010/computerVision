import mediapipe as mp
from detections import *
from rendering import *
from config import *


bow_img = cv2.imread('static/images/bow.png', cv2.IMREAD_UNCHANGED)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

with (mp_pose.Pose(min_detection_confidence=DETECTION_COEF) as pose_detector,
      mp_hands.Hands(min_detection_confidence=DETECTION_COEF) as hand_detector):
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

        pose_results = pose_detector.process(img_rgb)
        hand_results = hand_detector.process(img_rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            left_knuckle = landmarks[20]
            left_wrist = landmarks[16]

            if hand_results.multi_hand_landmarks is not None and detect_fist(res_image, hand_results):
                draw_bow(res_image, bow_img, left_knuckle.x, left_knuckle.y, left_wrist.x, left_wrist.y - WRIST_SHIFT)

        cv2.imshow(WIN_NAME, res_image)

        if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
