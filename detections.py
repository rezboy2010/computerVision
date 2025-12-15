from config import *
import numpy as np
import cv2


def get_points(landmark, shape):
    points = []

    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])

    return np.array(points, dtype=np.int32)


def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2)**2 + (y1 - y2) ** 2) ** .5


def detect_fist(img, results):
    (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, img.shape))
    ws = palm_size(results.multi_hand_landmarks[0].landmark, img.shape)
    return 2 * r / ws <= FIST_THRESHOLD
