from __future__ import annotations

import os
import re
import sys
from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_features import (FEATURE_NAMES, _segment_lines, compute_features,  # noqa: E402
                            extract_ink)

CANVAS_H, CANVAS_W = 64, 256
U_OUT = 12.0          # letter height (px) on the canvas
LEFT_PAD = 2
STROKE_GAIN = 1.8     # keeps thin strokes visible after down-scaling

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLASSES = {"potential_dysgraphia": 1, "low_potential_dysgraphia": 0}
N_AUG = 6


def ink_to_canvas(ink: np.ndarray, u: float) -> np.ndarray:
    """Binary ink mask (+ letter unit ``u``) -> uint8 (CANVAS_H, CANVAS_W)."""
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    s = min(U_OUT / max(u, 1.0), CANVAS_H / h, (CANVAS_W - LEFT_PAD) / w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    small = np.clip(small * STROKE_GAIN, 0.0, 1.0)
    canvas = np.zeros((CANVAS_H, CANVAS_W), np.float32)
    y0 = (CANVAS_H - nh) // 2
    canvas[y0:y0 + nh, LEFT_PAD:LEFT_PAD + nw] = small[:CANVAS_H, :CANVAS_W - LEFT_PAD]
    return (canvas * 255).astype(np.uint8)


LINES_PER_CHUNK = 2
MIN_CHUNK_INK = 0.8   # in letter-areas (u*u); smaller pieces are noise / stray marks


def _split_wide(piece, u):
    """Cut a very wide strip into pieces that each fit the canvas at letter size U_OUT."""
    max_w = (CANVAS_W - LEFT_PAD) * u / U_OUT
    w = piece.shape[1]
    if w <= 1.2 * max_w:
        return [piece]
    n = int(np.ceil(w / max_w))
    occ = np.convolve(piece.sum(axis=0).astype(float), np.ones(5) / 5, mode="same")
    cuts = [0]
    for k in range(1, n):
        target = int(k * w / n)
        lo, hi = max(cuts[-1] + 1, int(target - 0.15 * max_w)), min(w - 1, int(target + 0.15 * max_w))
        cuts.append(int(lo + np.argmin(occ[lo:hi + 1])) if hi > lo else target)
    cuts.append(w)
    return [piece[:, a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def ink_to_chunks(ink, comps, u):
    """Split the writing into text-line chunks (<= LINES_PER_CHUNK lines, <= canvas width).

    A whole page of text must not be squeezed onto one small canvas (letters would
    become unreadable and results would depend on how many lines were photographed),
    so every chunk is classified separately and the results are combined."""
    lines = _segment_lines(ink, comps, u)
    keep = [l for l in lines if len(l) >= 3] or lines
    pieces = []
    for i in range(0, len(keep), LINES_PER_CHUNK):
        idx = np.concatenate(keep[i:i + LINES_PER_CHUNK])
        c = comps[idx]
        y0, y1 = int(c[:, 1].min()), int((c[:, 1] + c[:, 3]).max()) + 1
        strip = ink[y0:y1]
        cols = np.nonzero(strip.any(axis=0))[0]
        if len(cols) == 0:
            continue
        pieces += _split_wide(strip[:, cols.min():cols.max() + 1], u)
    pieces = [p for p in pieces if p.sum() >= MIN_CHUNK_INK * u * u]
    if not pieces:
        pieces = [ink]
    return [ink_to_canvas(p, u) for p in pieces]


def image_to_canvases(source):
    """Any image source -> (list of chunk canvases uint8, feature dict).
    Raises NoHandwritingError."""
    ink, comps, u, _ = extract_ink(source)
    feats = compute_features(ink, comps, u)
    for k in FEATURE_NAMES:
        if not np.isfinite(feats[k]):
            feats[k] = 0.0
    return ink_to_chunks(ink, comps, u), feats


def image_to_canvas(source):
    """Single-canvas variant (whole image on one canvas); kept for tests/tools."""
    ink, comps, u, _ = extract_ink(source)
    feats = compute_features(ink, comps, u)
    return ink_to_canvas(ink, u), feats


# --------------------------------------------------------------------------
# Training-set cache (clean original + N_AUG synthetic photographs per image)
# --------------------------------------------------------------------------
def _one(path, label, seed):
    from augment import render_as_photo
    n = int(re.search(r"\((\d+)\)", path).group(1))
    gray = np.array(Image.open(path).convert("L"))
    rng = np.random.default_rng(seed)
    metas, canvases = [], []
    for v in range(N_AUG + 1):
        base = dict(label=label, child=f"{label}_{(n - 1) // 3}",
                    img=os.path.basename(path), variant=v)
        try:
            src = path if v == 0 else render_as_photo(gray, rng)
            chunks, feats = image_to_canvases(src)
            for ci, canvas in enumerate(chunks):
                metas.append({**feats, **base, "chunk": ci})
                canvases.append(canvas)
        except Exception as exc:                              # keep going, report later
            metas.append({**base, "chunk": 0, "error": str(exc)})
            canvases.append(np.zeros((CANVAS_H, CANVAS_W), np.uint8))
    return metas, canvases


def build_cache(features_csv, npz_path):
    jobs = []
    for cls, lab in CLASSES.items():
        paths = sorted(glob(os.path.join(ROOT, "data", "raw", cls, "*.*")))
        paths = [p for p in paths if re.search(r"\((\d+)\)", p)]
        print(f"  {cls}: {len(paths)} images")
        jobs += [(p, lab, 1000 * lab + i) for i, p in enumerate(paths)]
    if not jobs:
        raise RuntimeError("No training images found in data/raw/<class>/")
    rows, canv = [], []
    for i, j in enumerate(jobs):
        m, c = _one(*j)
        rows += m
        canv += c
        if (i + 1) % 25 == 0:
            print(f"  prepared {i + 1}/{len(jobs)} images", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(features_csv), exist_ok=True)
    df.to_csv(features_csv, index=False)          # row i  <->  X[i]
    np.savez_compressed(npz_path, X=np.stack(canv))
    return df, np.stack(canv)


def load_cache(rebuild=False):
    features_csv = os.path.join(ROOT, "data", "processed", "features.csv")
    npz_path = os.path.join(ROOT, "data", "processed", "dl_canvas.npz")
    if rebuild or not (os.path.exists(features_csv) and os.path.exists(npz_path)):
        return build_cache(features_csv, npz_path)
    print(f"  using cached data: {npz_path}")
    return pd.read_csv(features_csv), np.load(npz_path)["X"]
