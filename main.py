import mediapipe as mp
from detections import *
from rendering import *
from config import *
import time


bow_img = cv2.imread('static/images/bow.png', cv2.IMREAD_UNCHANGED)
arrow_img = cv2.imread('static/images/arrow.png', cv2.IMREAD_UNCHANGED)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
was_hand_behind_back = False
is_arrow_in_hand = False
is_bow_drawn = False

last_fist_t = 0.0

fist_on = fist_off = 0
fist_state = False

alpha = 0.35
lk_prev = lw_prev = None

with (mp_pose.Pose(min_detection_confidence=POSE_COEF, min_tracking_confidence=0.6) as pose_detector,
      mp_hands.Hands(min_detection_confidence=HAND_COEF, min_tracking_confidence=0.65) as hand_detector):
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

            hand_behind = is_hand_behind_back(landmarks)

            if was_hand_behind_back and not hand_behind:
                is_arrow_in_hand = not is_arrow_in_hand

            was_hand_behind_back = hand_behind

            t = time.time()

            raw_fist_now = (hand_results.multi_hand_landmarks is not None) and detect_fist(res_image, hand_results)

            if raw_fist_now:
                last_fist_t = t
            raw_fist = raw_fist_now or (t - last_fist_t) < FIST_HOLD_S

            # сколько кадров подрят показывается или не показывается
            if raw_fist:
                fist_on += 1
                fist_off = 0
            else:
                fist_off += 1
                fist_on = 0

            if not fist_state and fist_on >= 2: # включение
                fist_state = True
            elif fist_state and fist_off >= 8:  # выключение
                fist_state = False

            if fist_state:
                lk = (left_knuckle.x, left_knuckle.y)
                lw = (left_wrist.x, left_wrist.y)

                if lk_prev is None:
                    lk_prev, lw_prev = lk, lw
                else:
                    lk_prev = (alpha * lk[0] + (1 - alpha) * lk_prev[0], alpha * lk[1] + (1 - alpha) * lk_prev[1])
                    lw_prev = (alpha * lw[0] + (1 - alpha) * lw_prev[0], alpha * lw[1] + (1 - alpha) * lw_prev[1])

                draw_bow(res_image, bow_img, lk_prev[0], lk_prev[1], lw_prev[0], lw_prev[1])

            if is_arrow_in_hand:
                draw_bow(res_image, arrow_img, landmarks[19].x, landmarks[19].y, landmarks[15].x, landmarks[15].y)

        cv2.imshow(WIN_NAME, res_image)

        if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
