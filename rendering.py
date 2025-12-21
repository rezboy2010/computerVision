from config import *
import numpy as np
import cv2


def draw_bow(img, bow, knuckle_x: int, knuckle_y: int, wrist_x: int, wrist_y: int):
    wrist_y -= WRIST_SHIFT

    # Была подборка коэф-ов для поворота
    dx = wrist_x - knuckle_x
    dy = wrist_y - knuckle_y - 0.05
    angle_rad = np.arctan2(dy, dx) * 1.1
    angle_deg = -angle_rad * 180.0 / np.pi + 35

    BOW_H, BOW_W = bow.shape[:2]
    center = (BOW_W // 2, BOW_H // 2)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_bow = cv2.warpAffine(
        bow,
        M,
        (BOW_W, BOW_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    bow = rotated_bow
    h, w, _ = img.shape
    wx = int(knuckle_x * w)
    wy = int(knuckle_y * h)

    x0 = wx - BOW_W // 2
    y0 = wy - BOW_H // 2
    x1 = wx + BOW_W // 2
    y1 = wy + BOW_H // 2

    x0_cl = max(0, x0)
    y0_cl = max(0, y0)
    x1_cl = min(w, x1)
    y1_cl = min(h, y1)

    output_w = x1_cl - x0_cl
    output_h = y1_cl - y0_cl

    if output_w > 0 and output_h > 0:
        bx0 = x0_cl - x0
        by0 = y0_cl - y0
        bx1 = bx0 + output_w
        by1 = by0 + output_h

        bow_roi = bow[by0:by1, bx0:bx1]
        img_roi = img[y0_cl:y1_cl, x0_cl:x1_cl]

        b_bow, g_bow, r_bow, a_bow = cv2.split(bow_roi)
        alpha = a_bow.astype(float) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])

        bow_rgb = cv2.merge([b_bow, g_bow, r_bow]).astype(float)
        img_roi = img_roi.astype(float)

        blended = alpha * bow_rgb + (1 - alpha) * img_roi
        img[y0_cl:y1_cl, x0_cl:x1_cl] = blended.astype(np.uint8)
