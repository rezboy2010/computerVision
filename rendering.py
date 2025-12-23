from config import *
import numpy as np
import cv2


def draw(img, obj, knuckle_x: int, knuckle_y: int,
         wrist_x: int = None, wrist_y: int = None, angle_deg: float = None, arrow: bool = False):
    if angle_deg is None:
        # Была подборка коэф-ов для поворота
        wrist_y -= WRIST_SHIFT
        dx = wrist_x - knuckle_x
        dy = wrist_y - knuckle_y - 0.05
        angle_rad = np.arctan2(dy, dx) * 1.1
        angle_deg = -angle_rad * 180.0 / np.pi + 35

    if arrow:
        knuckle_y += np.sin(np.radians(angle_deg)) * ARROW_LENGTH

    OBJ_H, OBJ_W = obj.shape[:2]
    center = (OBJ_W // 2, OBJ_H // 2)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_obj = cv2.warpAffine(
        obj,
        M,
        (OBJ_W, OBJ_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    obj = rotated_obj
    h, w, _ = img.shape
    wx = int(knuckle_x * w)
    wy = int(knuckle_y * h)

    x0 = wx - OBJ_W // 2
    y0 = wy - OBJ_H // 2
    x1 = wx + OBJ_W // 2
    y1 = wy + OBJ_H // 2

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

        bow_roi = obj[by0:by1, bx0:bx1]
        img_roi = img[y0_cl:y1_cl, x0_cl:x1_cl]

        b_bow, g_bow, r_bow, a_bow = cv2.split(bow_roi)
        alpha = a_bow.astype(float) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])

        bow_rgb = cv2.merge([b_bow, g_bow, r_bow]).astype(float)
        img_roi = img_roi.astype(float)

        blended = alpha * bow_rgb + (1 - alpha) * img_roi
        img[y0_cl:y1_cl, x0_cl:x1_cl] = blended.astype(np.uint8)

    return angle_deg


def bow_tips_frame(frame, bow_img, kn_x, kn_y, angle_deg, tips_local):
    H, W = bow_img.shape[:2]
    h, w = frame.shape[:2]
    cx, cy = W/2, H/2

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += kn_x * w - cx
    M[1, 2] += kn_y * h - cy

    def tr(p):
        x, y = p
        X = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        Y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return int(round(X)), int(round(Y))

    return tr(tips_local[0]), tr(tips_local[1])


STATE_MENU = "menu"
STATE_GAME = "game"

btn_play = (SCREEN_W // 2 - 100, 200, SCREEN_W // 2 + 125, 300)


def in_rect(x, y, r):
    x1, y1, x2, y2 = r
    return x1 <= x <= x2 and y1 <= y <= y2


cv2.namedWindow(WIN_NAME)


def draw_button(img, rect, text):
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(img, text, (x1 + 20, y1 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2, cv2.LINE_AA)


def render_menu(img):
    cv2.putText(img, "Arrow of Fate", (SCREEN_W // 2 - 300, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (140, 230, 240), 3, cv2.LINE_AA)
    draw_button(img, btn_play, "START")


def render_results(img, score, best_score):
    cv2.putText(img, f'Your score: {score:.2f}', (700, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (144, 128, 112), 2, cv2.LINE_AA)
    cv2.putText(img, f'Previos best score: {best_score:.2f}', (600, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (11, 134, 184), 2, cv2.LINE_AA)
    draw_button(img, btn_play, "Again")
