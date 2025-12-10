def draw_bow(img, bow, x: int, y: int):
    h, w, _ = img.shape
    BOW_H, BOW_W = bow.shape[:2]
    wx = int(x * w)
    wy = int(y * h)

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

        img[y0_cl:y1_cl, x0_cl:x1_cl] = bow[by0:by1, bx0:bx1]
