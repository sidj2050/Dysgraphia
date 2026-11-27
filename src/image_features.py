from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

FEATURE_NAMES = [
    "ink_ratio", "foreground_bbox_width_ratio", "foreground_bbox_height_ratio",
    "foreground_aspect_ratio", "connected_component_count", "component_area_mean",
    "component_area_std", "component_area_cv", "component_width_mean",
    "component_width_std", "component_height_mean", "component_height_std",
    "component_aspect_mean", "component_aspect_std", "component_centroid_y_std",
    "horizontal_projection_mean", "horizontal_projection_std", "vertical_projection_mean",
    "vertical_projection_std", "row_ink_cv", "col_ink_cv",
]

TARGET_U = 40.0      # letter height (px) the image is normalised to
MAX_WORK = 1400      # first-pass working size (longest side)
MIN_COMPONENTS = 6   # fewer than this => "no handwriting found"


class NoHandwritingError(ValueError):
    """Raised when the image does not contain enough handwriting to analyse."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _pil_to_rgb(im):
    im = ImageOps.exif_transpose(im)          # phone photos carry rotation in EXIF
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    if im.mode in ("I;16", "I"):
        arr = np.array(im, dtype=np.float32)
        arr = (255 * (arr - arr.min()) / max(arr.max() - arr.min(), 1)).astype(np.uint8)
        im = Image.fromarray(arr)
    return np.array(im.convert("RGB"), dtype=np.uint8)


def load_rgb(source):
    """Decode a path, bytes, PIL image or numpy array (BGR/gray) into RGB uint8."""
    if isinstance(source, (str, Path)):
        try:
            with Image.open(source) as im:
                im.load()
                return _pil_to_rgb(im)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            img = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not decode image: {exc}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(source, (bytes, bytearray, memoryview)):
        try:
            with Image.open(io.BytesIO(bytes(source))) as im:
                im.load()
                return _pil_to_rgb(im)
        except Exception as exc:
            raise ValueError(f"Could not decode image bytes: {exc}")
    if hasattr(source, "convert") and hasattr(source, "size"):
        return _pil_to_rgb(source)
    if isinstance(source, np.ndarray):
        if source.size == 0:
            raise ValueError("The supplied image array is empty.")
        a = source
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            return np.stack([a] * 3, axis=-1)
        if a.ndim == 3 and a.shape[2] == 1:
            return np.repeat(a, 3, axis=2)
        if a.ndim == 3 and a.shape[2] == 3:      # OpenCV convention: BGR
            return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        if a.ndim == 3 and a.shape[2] == 4:
            return cv2.cvtColor(a, cv2.COLOR_BGRA2RGB)
        raise ValueError(f"Unsupported image array shape: {a.shape}")
    raise TypeError(f"Unsupported image source type: {type(source).__name__}")


# --------------------------------------------------------------------------
# Binarisation / cleaning
# --------------------------------------------------------------------------
def _resize(img, scale):
    if abs(scale - 1.0) < 1e-3:
        return img
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


def _ink_strength(gray):
    """Illumination-flattened 'how much darker than the paper' map (uint8)."""
    h, w = gray.shape
    k = int(max(31, min(81, 0.04 * max(h, w)))) | 1
    bg = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(bg, k)
    return cv2.subtract(bg, gray)


def _threshold(diff, T=None):
    if T is None:
        otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        T = max(28.0, 0.85 * otsu)
    return (diff > T).astype(np.uint8), T


def _rotate(img, angle):
    if abs(angle) < 0.15:
        return img
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - c[0]
    M[1, 2] += nh / 2.0 - c[1]
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)


def _estimate_skew(mask):
    """Angle (deg) that makes text/ruled lines horizontal (projection sharpness)."""
    h, w = mask.shape
    s = 360.0 / max(h, w)
    small = cv2.resize(mask.astype(np.float32), None, fx=s, fy=s,
                       interpolation=cv2.INTER_AREA) if s < 1 else mask.astype(np.float32)
    H, W = small.shape
    bh, bw = max(2, int(0.04 * H)), max(2, int(0.04 * W))   # ignore photo-edge artefacts
    small = small.copy()
    small[:bh], small[-bh:], small[:, :bw], small[:, -bw:] = 0, 0, 0, 0
    D = int(np.ceil(np.hypot(H, W))) + 2
    canvas = np.zeros((D, D), np.float32)
    oy, ox = (D - H) // 2, (D - W) // 2
    canvas[oy:oy + H, ox:ox + W] = small

    def score(a):
        M = cv2.getRotationMatrix2D((D / 2.0, D / 2.0), a, 1.0)
        r = cv2.warpAffine(canvas, M, (D, D), flags=cv2.INTER_LINEAR)
        rows = r.sum(axis=1)
        return float(np.sum(np.diff(rows) ** 2))

    coarse = np.arange(-50.0, 50.1, 2.0)
    sc = np.array([score(a) for a in coarse])
    # prior: photographed pages are usually close to upright
    sc = sc * (1.0 - 0.004 * np.abs(coarse))
    best = float(coarse[int(np.argmax(sc))])
    fine = np.arange(best - 2.0, best + 2.01, 0.25)
    fs = np.array([score(a) for a in fine])
    return float(fine[int(np.argmax(fs))])


def _remove_ruled_lines(ink):
    """Delete long horizontal rules but keep strokes that cross them."""
    h, w = ink.shape
    L = int(np.clip(0.10 * w, 90, 220))
    core = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    if core.sum() == 0:
        return ink
    linemask = cv2.dilate(core, np.ones((5, 3), np.uint8)) & ink
    rest = ink & (1 - linemask)
    above = np.zeros_like(rest)
    below = np.zeros_like(rest)
    d = 5
    for k in range(1, d + 1):
        above[k:] |= rest[:-k]       # pixel has ink k rows above it
        below[:-k] |= rest[k:]       # pixel has ink k rows below it
    bridge = linemask & above & below
    return rest | bridge


def _components(ink, u=None):
    h, w = ink.shape
    n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(12, int(1.5e-5 * h * w))
    cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not cand:
        return np.zeros((0, 7)), np.zeros(n, bool), lab
    med_h = float(np.median(stats[cand, cv2.CC_STAT_HEIGHT]))
    keep = np.zeros(n, bool)
    for i in cand:
        x, y, cw, ch, a = stats[i]
        longs, shorts = max(cw, ch), max(1, min(cw, ch))
        if longs > 4.0 * med_h and longs / shorts >= 10:        # leftover rules / page edge
            continue
        if a > 0.25 * h * w:                                   # shadow / blob
            continue
        if u is not None and ch < 0.2 * u and cw / max(ch, 1) > 5:   # flat rule fragments
            continue
        keep[i] = True
    rows = [(*stats[i, :4], stats[i, 4], cents[i][0], cents[i][1]) for i in np.where(keep)[0]]
    return np.array(rows, dtype=float).reshape(-1, 7), keep, lab


def _cut(ink, mask, axis):
    """Remove masked line pixels but keep strokes that cross the line."""
    lm = mask & ink
    rest = ink & (1 - lm)
    a, b, d = np.zeros_like(rest), np.zeros_like(rest), 5
    for k in range(1, d + 1):
        if axis == 0:
            a[k:] |= rest[:-k]; b[:-k] |= rest[k:]
        else:
            a[:, k:] |= rest[:, :-k]; b[:, :-k] |= rest[:, k:]
    return rest | (lm & a & b)


def _remove_straight_rules(ink, u):
    """Remove long straight rules at any small tilt (margin lines, tilted rulings)."""
    segs = cv2.HoughLinesP(ink * 255, 1, np.pi / 180, threshold=int(1.5 * u),
                           minLineLength=int(2.8 * u), maxLineGap=max(2, int(0.2 * u)))
    if segs is None:
        return ink
    hm, vm = np.zeros_like(ink), np.zeros_like(ink)
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        dev = min(ang, 180 - ang)                 # deviation from horizontal
        if dev < 12:
            cv2.line(hm, (int(x1), int(y1)), (int(x2), int(y2)), 1, 5)
        elif dev > 60:
            cv2.line(vm, (int(x1), int(y1)), (int(x2), int(y2)), 1, 5)
    if hm.any():
        ink = _cut(ink, hm, 0)
    if vm.any():
        ink = _cut(ink, vm, 1)
    return ink


def _letter_unit(comps):
    areas = comps[:, 4]
    big = comps[areas >= np.median(areas)]
    return float(np.median(big[:, 3])) if len(big) else float(np.median(comps[:, 3]))


def _segment(gray, angle=None, T=None):
    diff = _ink_strength(gray)
    if diff.max() < 28:
        raise NoHandwritingError("No handwriting could be detected in this image.")
    ink0, T = _threshold(diff, T)
    if angle is None:
        angle = _estimate_skew(ink0)
    diff = _rotate(diff, angle)
    ink, _ = _threshold(diff, T)
    ink = _remove_ruled_lines(ink)
    comps, keep, lab = _components(ink)
    if len(comps) < MIN_COMPONENTS:
        raise NoHandwritingError("Not enough handwriting could be detected in this image.")
    clean = np.isin(lab, np.where(keep)[0]).astype(np.uint8)
    return clean, comps, angle, T


def extract_ink(source):
    """Return the canonical ink mask, component table, letter unit and skew angle."""
    rgb = load_rgb(source)
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = np.clip(0.5 * lum + 0.5 * rgb.min(axis=2).astype(np.float32), 0, 255).astype(np.uint8)
    if np.median(gray) < 100:                  # light ink on dark background -> invert
        gray = 255 - gray
    h0, w0 = gray.shape
    s1 = float(np.clip(MAX_WORK / max(h0, w0), 0.05, 4.0))
    g1 = _resize(gray, s1)
    ink, comps, angle, T = _segment(g1)
    u = _letter_unit(comps)
    s2 = float(np.clip(TARGET_U / max(u, 1.0), 0.25, 4.0))
    if not (0.85 <= s2 <= 1.18):               # re-run at the normalised scale
        g2 = _resize(gray, s1 * s2)
        ink, comps, _, _ = _segment(g2, angle=angle)
        u = _letter_unit(comps)
    # final cleanup now that the letter size is known
    ink = _remove_straight_rules(ink, u)
    comps, keep, lab = _components(ink, u)
    if len(comps) < MIN_COMPONENTS:
        raise NoHandwritingError("Not enough handwriting could be detected in this image.")
    ink = np.isin(lab, np.where(keep)[0]).astype(np.uint8)
    u = _letter_unit(comps)
    return ink, comps, max(u, 1.0), angle


# --------------------------------------------------------------------------
# Line segmentation + features
# --------------------------------------------------------------------------
def _segment_lines(ink, comps, u):
    prof = ink.sum(axis=1).astype(float)
    sm = gaussian_filter1d(prof, sigma=max(2.0, 0.3 * u))
    peaks, _ = find_peaks(sm, prominence=0.15 * sm.max(), distance=max(3, int(0.9 * u)))
    if len(peaks) <= 1:
        return [np.arange(len(comps))]
    bounds = [int(p0 + np.argmin(sm[p0:p1 + 1])) for p0, p1 in zip(peaks[:-1], peaks[1:])]
    idx = np.searchsorted(bounds, comps[:, 6])
    lines = [np.where(idx == k)[0] for k in range(len(peaks))]
    return [l for l in lines if len(l) > 0]


def _cv(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    return float(np.std(values) / (abs(np.mean(values)) + 1e-8))


def compute_features(ink, comps, u):
    w_, h_, area = comps[:, 2] / u, comps[:, 3] / u, comps[:, 4] / (u * u)
    aspect = np.clip(comps[:, 2] / np.maximum(comps[:, 3], 1.0), 0.05, 10.0)
    lines = _segment_lines(ink, comps, u)
    good = [l for l in lines if len(l) >= 3] or [np.arange(len(comps))]
    med_w = float(np.median(w_))

    per = {k: [] for k in ["w", "h", "asp", "dens", "adv", "cy", "ink",
                           "rmean", "rstd", "occ", "cstd", "rcv", "ccv"]}
    tot_ink = tot_region = 0.0
    for l in good:
        c = comps[l]
        x0, x1 = (c[:, 0]).min(), (c[:, 0] + c[:, 2]).max()
        y0, y1 = (c[:, 1]).min(), (c[:, 1] + c[:, 3]).max()
        ext_w, band_h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        region = ink[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
        row = region.mean(axis=1)
        col = region.mean(axis=0)
        per["w"].append(ext_w / (len(l) * med_w * u))
        per["h"].append(band_h / u)
        per["asp"].append(min(ext_w / band_h, 40.0))
        per["dens"].append(len(l) / (ext_w / (10.0 * u)))
        per["cy"].append(float(np.std(c[:, 6]) / u))
        per["rmean"].append(float(row.mean()))
        per["rstd"].append(float(row.std()))
        per["occ"].append(float((col > 0).mean()))
        per["cstd"].append(float(col.std()))
        per["rcv"].append(_cv(row))
        per["ccv"].append(_cv(col))
        tot_ink += float(region.sum())
        tot_region += float(region.size)

    med = lambda k: float(np.median(per[k]))
    return {
        "ink_ratio": tot_ink / max(tot_region, 1.0),
        "foreground_bbox_width_ratio": med("w"),
        "foreground_bbox_height_ratio": med("h"),
        "foreground_aspect_ratio": med("asp"),
        "connected_component_count": med("dens"),
        "component_area_mean": float(np.mean(area)),
        "component_area_std": float(np.std(area)),
        "component_area_cv": _cv(area),
        "component_width_mean": float(np.mean(w_)),
        "component_width_std": float(np.std(w_)),
        "component_height_mean": float(np.mean(h_)),
        "component_height_std": float(np.std(h_)),
        "component_aspect_mean": float(np.mean(aspect)),
        "component_aspect_std": float(np.std(aspect)),
        "component_centroid_y_std": med("cy"),
        "horizontal_projection_mean": med("rmean"),
        "horizontal_projection_std": med("rstd"),
        "vertical_projection_mean": med("occ"),
        "vertical_projection_std": med("cstd"),
        "row_ink_cv": med("rcv"),
        "col_ink_cv": med("ccv"),
    }


def extract_features(source):
    """Return the numeric feature dictionary (21 characteristics) for one image."""
    ink, comps, u, _ = extract_ink(source)
    feats = compute_features(ink, comps, u)
    for k in FEATURE_NAMES:
        if not np.isfinite(feats[k]):
            feats[k] = 0.0
    return feats


# --------------------------------------------------------------------------
# Human-readable interpretation (labels unchanged from the original app)
# --------------------------------------------------------------------------
LABELS = {
    "ink_ratio": "Ink coverage",
    "foreground_bbox_width_ratio": "Writing width",
    "foreground_bbox_height_ratio": "Writing height",
    "foreground_aspect_ratio": "Writing aspect ratio",
    "connected_component_count": "Stroke/component count",
    "component_area_mean": "Average component size",
    "component_area_std": "Component size variation",
    "component_area_cv": "Component size consistency",
    "component_width_mean": "Average component width",
    "component_width_std": "Component width variation",
    "component_height_mean": "Average component height",
    "component_height_std": "Component height variation",
    "component_aspect_mean": "Component shape ratio",
    "component_aspect_std": "Component shape consistency",
    "component_centroid_y_std": "Vertical alignment variation",
    "horizontal_projection_mean": "Horizontal ink distribution",
    "horizontal_projection_std": "Horizontal ink variation",
    "vertical_projection_mean": "Vertical ink distribution",
    "vertical_projection_std": "Vertical ink variation",
    "row_ink_cv": "Row ink consistency",
    "col_ink_cv": "Column ink consistency",
}


def feature_interpretations(features, reference):
    """Compare each measured characteristic with the typical-writer reference range."""
    out = []
    for key in FEATURE_NAMES:
        value = features[key]
        ref = reference.get(key, {})
        p10, p90 = ref.get("p10"), ref.get("p90")
        if p10 is None or p90 is None:
            status, direction = "Typical", None
            reason = f"Measured value: {value:.3g}."
        elif value < p10 or value > p90:
            direction = "higher" if value > p90 else "lower"
            status = "Unusual"
            reason = (f"Measured {value:.3g} is {direction} than the typical range "
                      f"({p10:.3g}–{p90:.3g}).")
        else:
            direction, status = None, "Typical"
            reason = f"Measured {value:.3g} is within the typical range ({p10:.3g}–{p90:.3g})."
        out.append({"feature": LABELS.get(key, key), "status": status,
                    "reason": reason, "key": key, "value": value, "direction": direction})
    return out
