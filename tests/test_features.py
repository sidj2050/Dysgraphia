import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from image_features import FEATURE_NAMES, NoHandwritingError, extract_features  # noqa: E402


def _page(dark_on_light=True, angle=0.0):
    img = np.full((500, 900), 255, np.uint8)
    for i, txt in enumerate(["Hello world again", "second line of text", "third line here"]):
        cv2.putText(img, txt, (40, 120 + 130 * i), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.2, 0, 4)
    if angle:
        M = cv2.getRotationMatrix2D((450, 250), angle, 1)
        img = cv2.warpAffine(img, M, (900, 500), borderValue=255)
    return img if dark_on_light else 255 - img


def test_all_21_features_finite():
    f = extract_features(_page())
    assert list(f.keys()) == FEATURE_NAMES and len(f) == 21
    assert all(np.isfinite(v) for v in f.values())


def test_polarity_invariant():
    a, b = extract_features(_page(True)), extract_features(_page(False))
    for k in FEATURE_NAMES:
        assert a[k] == pytest.approx(b[k], rel=0.05, abs=0.02)


def test_rotation_robust():
    a, b = extract_features(_page()), extract_features(_page(angle=8))
    assert a["component_height_mean"] == pytest.approx(b["component_height_mean"], rel=0.25)


def test_blank_image_rejected():
    with pytest.raises(NoHandwritingError):
        extract_features(np.full((300, 600), 255, np.uint8))
