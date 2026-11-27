import cv2
import numpy as np

INK_COLOURS = [(18, 18, 18), (55, 55, 60), (35, 45, 150), (20, 30, 105),
               (25, 25, 25), (150, 30, 40), (90, 90, 96), (30, 60, 140)]


def _smooth_field(h, w, rng, lo, hi):
    """Low-frequency multiplicative lighting field in [lo, hi]."""
    g = rng.uniform(size=(4, 4)).astype(np.float32)
    f = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    f = (f - f.min()) / max(f.max() - f.min(), 1e-6)
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (np.cos(ang) * xx / max(w, 1) + np.sin(ang) * yy / max(h, 1))
    grad = (grad - grad.min()) / max(grad.max() - grad.min(), 1e-6)
    mix = 0.5 * f + 0.5 * grad
    return lo + (hi - lo) * mix


def render_as_photo(img_gray_white_on_black, rng):
    ink = img_gray_white_on_black.astype(np.float32) / 255.0
    if ink.mean() > 0.5:                      # safety: make ink the minority
        ink = 1.0 - ink

    # --- resolution / zoom -------------------------------------------------
    s = float(np.exp(rng.uniform(np.log(0.45), np.log(1.9))))
    ink = cv2.resize(ink, None, fx=s, fy=s,
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    ink = np.clip(ink, 0, 1)

    # --- occasional crop (photos often cut a line or the page edge) --------
    h, w = ink.shape
    if rng.uniform() < 0.25 and w > 300:
        keep = rng.uniform(0.6, 0.95)
        x0 = int(rng.uniform(0, 1 - keep) * w)
        ink = ink[:, x0:x0 + int(keep * w)]
        h, w = ink.shape

    # --- paper with margins -------------------------------------------------
    pad = lambda: int(rng.uniform(0.0, 0.5) * h) + int(rng.uniform(0, 40))
    t, b, l, r = pad(), pad(), pad(), pad()
    canvas = np.pad(ink, ((t, b), (l, r)), mode="constant")
    H, W = canvas.shape
    paper = np.array([rng.uniform(222, 255), rng.uniform(222, 255), rng.uniform(205, 255)],
                     dtype=np.float32)
    page = np.ones((H, W, 3), np.float32) * paper

    # --- ruled lines / margin line (drawn under the ink) -------------------
    if rng.uniform() < 0.65:
        spacing = int(rng.uniform(0.5, 1.4) * max(30, h / rng.uniform(1.0, 3.0)))
        spacing = max(spacing, 24)
        dark = rng.uniform() < 0.25
        col = (np.array([rng.uniform(30, 110)] * 3) if dark else
               np.array([rng.uniform(150, 215), rng.uniform(170, 225), rng.uniform(205, 250)]))
        if rng.uniform() < 0.3:
            col = np.array([rng.uniform(215, 250), rng.uniform(150, 190), rng.uniform(165, 205)])
        th = int(rng.integers(1, 4))
        y = int(rng.uniform(0, spacing))
        while y < H:
            page[y:y + th, :, :] = col
            y += spacing
        if rng.uniform() < 0.25:
            x = int(rng.uniform(0.02, 0.15) * W)
            page[:, x:x + 2, :] = np.array([220, 90, 100])

    # --- composite ink -------------------------------------------------------
    colour = np.array(INK_COLOURS[int(rng.integers(len(INK_COLOURS)))], dtype=np.float32)
    colour = np.clip(colour + rng.normal(0, 8, 3), 0, 255)
    opacity = rng.uniform(0.75, 1.0)
    a = (canvas * opacity)[..., None]
    out = page * (1 - a) + colour * a

    # --- lighting / shadows -------------------------------------------------
    out *= _smooth_field(H, W, rng, rng.uniform(0.62, 0.95), 1.05)[..., None]

    # --- geometry: rotation + perspective ----------------------------------
    if rng.uniform() < 0.15:
        angle = rng.uniform(-45, 45)
    else:
        angle = float(np.clip(rng.normal(0, 6), -20, 20))
    c = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(H * sin + W * cos), int(H * cos + W * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    out = cv2.warpAffine(out, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    if rng.uniform() < 0.4:
        h2, w2 = out.shape[:2]
        j = lambda: rng.uniform(0, 0.06)
        src = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        dst = np.float32([[j() * w2, j() * h2], [w2 - j() * w2, j() * h2],
                          [w2 - j() * w2, h2 - j() * h2], [j() * w2, h2 - j() * h2]])
        P = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, P, (w2, h2), borderMode=cv2.BORDER_REPLICATE)

    # --- sensor: blur, noise, JPEG -----------------------------------------
    out = np.clip(out, 0, 255)
    if rng.uniform() < 0.7:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.3))
    out = out + rng.normal(0, rng.uniform(1, 7), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    q = int(rng.integers(40, 96))
    ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(out)
