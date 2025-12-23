from config import *
import numpy as np
import cv2
import time


def get_points(landmark, shape):
    points = []

    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])

    return np.array(points, dtype=np.int32)


def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5


def detect_fist(img, results, fist_threshold, hand: str, min_hand_score=0.7):
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return False

    idx = None
    for i, h in enumerate(results.multi_handedness):
        cls = h.classification[0]
        if cls.label == hand and cls.score >= min_hand_score:
            idx = i
            break
    if idx is None:
        return False

    lm = results.multi_hand_landmarks[idx].landmark
    _, r = cv2.minEnclosingCircle(get_points(lm, img.shape))
    ws = palm_size(lm, img.shape)
    return (2 * r / ws) <= fist_threshold


def is_hand_behind_back(landmarks):
    return landmarks[19].y > landmarks[13].y and landmarks[11].y > landmarks[13].y


def check_bow_drawn(bow_x, bow_y, arrow_x, arrow_y):
    return ((bow_x - arrow_x) ** 2 + (bow_y - arrow_y) ** 2) ** 0.5 < START_DIST


class Updater:
    def __init__(self, hand, fist_threshold):
        self.hand_results = None
        self.res_image = None
        self.hand = hand
        self.last_fist_t = 0.0
        self.fist_on = 0
        self.fist_off = 0
        self.fist_state = False
        self.fist_threshold = fist_threshold

    def update(self, hand_results, res_image):
        self.hand_results = hand_results
        self.res_image = res_image
        t = time.time()

        raw_fist_now = (self.hand_results.multi_hand_landmarks is not None) and detect_fist(self.res_image, self.hand_results, self.fist_threshold, self.hand)

        if raw_fist_now:
            self.last_fist_t = t

        raw_fist = raw_fist_now or (t - self.last_fist_t) < FIST_HOLD_S

        # сколько кадров подрят показывается или не показывается
        if raw_fist:
            self.fist_on += 1
            self.fist_off = 0
        else:
            self.fist_off += 1
            self.fist_on = 0

        if not self.fist_state and self.fist_on >= 2:  # включение
            self.fist_state = True
        elif self.fist_state and self.fist_off >= 3:  # выключение
            self.fist_state = False
