import mediapipe as mp
from detections import *
from rendering import *
from config import *
import winsound
import random


def play_one_shot():
    winsound.PlaySound('static/sounds/one_shot.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)


def play_results():
    winsound.PlaySound('static/sounds/finish_music.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)


def on_mouse(event, x, y, flags, param):
    global state

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state == STATE_MENU and in_rect(x, y, btn_play):
        state = STATE_GAME
    elif state == 'RESULTS' and in_rect(x, y, btn_play):
        state = STATE_GAME
        first_results = True


bow_img = cv2.imread('static/images/bow.png', cv2.IMREAD_UNCHANGED)
bow_img = cv2.resize(bow_img, (225 * 2, 245 * 2), interpolation=cv2.INTER_LINEAR)
arrow_img = cv2.imread('static/images/arrow.png', cv2.IMREAD_UNCHANGED)
arrow_img = cv2.resize(arrow_img, (201 * 2, 201 * 2), interpolation=cv2.INTER_LINEAR)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
was_hand_behind_back = False
is_arrow_in_hand = False
is_bow_drawn = False

alpha = 0.35
lk_prev = lw_prev = None
tips_local_b = ((131 * 2, 15 * 2), (131 * 2, 229 * 2))
tips_local_a = ((176 * 2, 98 * 2), (176 * 2, 100 * 2))

bow_updater = Updater('Left', 1.4)
arrow_updater = Updater('Right', 1.8)
arrow_flying = False
arrow_grace = 0
ARROW_GRACE_FRAMES = 10

target_x = 6
target_y1 = target_y2 = None
target_flag = True
state = STATE_MENU

first_time = True

with (mp_pose.Pose(min_detection_confidence=POSE_COEF, min_tracking_confidence=0.6) as pose_detector,
      mp_hands.Hands(min_detection_confidence=HAND_COEF, min_tracking_confidence=0.65) as hand_detector):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, SCREEN_W, SCREEN_H)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_image = frame.copy()
        res_image = cv2.resize(res_image, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)
        h, w = res_image.shape[:2]

        if state == STATE_MENU:
            render_menu(res_image)
        elif state == STATE_GAME:
            if first_time:
                winsound.PlaySound(None, winsound.SND_PURGE)
                start = time.time()
                first_time = False
                first_results = True
                score = 0
                counter = 0

            if 60 - time.time() + start < 0:
                state = 'RESULTS'

            cv2.putText(res_image, f"Score: {score:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2,
                        cv2.LINE_AA)
            cv2.putText(res_image, f"Time left: {60 - time.time() + start:.2f}", (800, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2,
                        cv2.LINE_AA)

            if target_y1 is None:
                shift = random.randint(50, 100) * 2
                shift = min(shift, h - 40)
                target_y1 = random.randint(20, h - shift - 400)
                target_y2 = target_y1 + shift

            pose_results = pose_detector.process(img_rgb)
            hand_results = hand_detector.process(img_rgb)
            cv2.line(res_image, (target_x, target_y1), (target_x, target_y2), (34, 34, 178), 12, cv2.LINE_AA)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                left_knuckle = landmarks[20]
                left_wrist = landmarks[16]

                bow_updater.update(hand_results, res_image)
                arrow_updater.update(hand_results, res_image)
                fist_state = bow_updater.fist_state
                arrow_state = arrow_updater.fist_state

                hand_behind = is_hand_behind_back(landmarks)

                if was_hand_behind_back and not hand_behind:
                    is_arrow_in_hand = not is_arrow_in_hand

                    if is_arrow_in_hand:
                        arrow_grace = ARROW_GRACE_FRAMES

                was_hand_behind_back = hand_behind

                if fist_state:
                    lk = (left_knuckle.x, left_knuckle.y)
                    lw = (left_wrist.x, left_wrist.y)

                    if lk_prev is None:
                        lk_prev, lw_prev = lk, lw
                    else:
                        lk_prev = (alpha * lk[0] + (1 - alpha) * lk_prev[0], alpha * lk[1] + (1 - alpha) * lk_prev[1])
                        lw_prev = (alpha * lw[0] + (1 - alpha) * lw_prev[0], alpha * lw[1] + (1 - alpha) * lw_prev[1])

                    angle_deg = draw(res_image, bow_img, lk_prev[0], lk_prev[1], lw_prev[0], lw_prev[1])

                if fist_state and is_arrow_in_hand and arrow_state and check_bow_drawn(lk_prev[0], lw_prev[1], landmarks[19].x, landmarks[19].y):
                    is_bow_drawn = True
                elif not fist_state:
                    is_bow_drawn = False

                if arrow_flying:
                    arrow_x += vx
                    arrow_y += vy
                    vy += g

                    if abs(vx) + abs(vy) > 1e-6:
                        ang = -np.degrees(np.arctan2(vy, vx)) + 180
                    else:
                        ang = angle_deg

                    draw(res_image, arrow_img, arrow_x, arrow_y, angle_deg=ang)
                    border_y = (arrow_y - ARROW_LENGTH * np.sin(np.radians(ang))) * h

                    if (ARROW_LENGTH * np.cos(np.radians(ang) > arrow_x) and target_flag and (target_y1 <= border_y <= target_y2)):
                        play_one_shot()
                        score += 5 * min(border_y - target_y1, target_y2 - border_y) / ((target_y2 - target_y1) / 2)

                        shift = random.randint(50, 150) * 2
                        shift = min(shift, h - 40)
                        target_y1 = random.randint(20, h - shift - 400)
                        target_y2 = target_y1 + shift

                        target_flag = False

                    if not (0.0 <= arrow_x <= 1.0 and 0.0 <= arrow_y <= 1.0):
                        arrow_flying = False
                        target_flag = True

                row_arrow_x = landmarks[19].x + ARROW_SHIFT

                # отрисовка тетивы
                if is_bow_drawn and not arrow_state:
                    is_bow_drawn = False
                    arrow_x = min(max(row_arrow_x, lk_prev[0] - MIN_STRETCHING * np.cos(np.radians(angle_deg))),
                                  lk_prev[0] + MAX_STRETCHING * np.cos(np.radians(angle_deg)))
                    arrow_y = lk_prev[1] + (lk_prev[0] - arrow_x) * np.tan(np.radians(angle_deg))
                    speed = (arrow_x - lk_prev[0] + MIN_STRETCHING * np.cos(np.radians(angle_deg))) / (MAX_STRETCHING + MIN_STRETCHING) / np.cos(np.radians(angle_deg))
                    a = np.radians(angle_deg)
                    base = 0.02 + 0.1 * speed

                    vx = -base * np.cos(a)
                    vy = base * np.sin(a)

                    arrow_flying = True
                elif is_bow_drawn:
                    p1, p2 = bow_tips_frame(res_image, bow_img, lk_prev[0], lk_prev[1], angle_deg, tips_local_b)
                    arrow_x = min(max(row_arrow_x, lk_prev[0] - MIN_STRETCHING * np.cos(np.radians(angle_deg))),
                                  lk_prev[0] + MAX_STRETCHING * np.cos(np.radians(angle_deg)))
                    arrow_y = lk_prev[1] + (lk_prev[0] - arrow_x) * np.tan(np.radians(angle_deg))
                    p3, p4 = bow_tips_frame(res_image, arrow_img, arrow_x, arrow_y, angle_deg, tips_local_a)
                    cv2.line(res_image, p1, p3, (79, 79, 47), 1, cv2.LINE_AA)
                    cv2.line(res_image, p3, p2, (79, 79, 47), 1, cv2.LINE_AA)
                elif fist_state:
                    p1, p2 = bow_tips_frame(res_image, bow_img, lk_prev[0], lk_prev[1], angle_deg, tips_local_b)
                    cv2.line(res_image, p1, p2, (79, 79, 47), 1, cv2.LINE_AA)

                if arrow_state and is_bow_drawn:
                    draw(res_image, arrow_img, arrow_x, arrow_y, angle_deg=angle_deg)
                elif arrow_state and is_arrow_in_hand:
                    draw(res_image, arrow_img, row_arrow_x, landmarks[19].y, landmarks[15].x + ARROW_SHIFT, landmarks[15].y, arrow=True)

                if arrow_grace > 0:
                    arrow_grace -= 1
                elif not arrow_state:
                    is_arrow_in_hand = False
        elif state == 'RESULTS' and first_results:
            play_results()

            with open('best_result.txt') as file:
                best_number = float(file.read())

            if score > best_number:
                with open('best_result.txt', mode='w') as file:
                    file.write(str(score))

            render_results(res_image, score, best_number)
            first_results = False
            first_time = True
        elif state == 'RESULTS':
            render_results(res_image, score, best_number)

        cv2.imshow(WIN_NAME, res_image)

        if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
