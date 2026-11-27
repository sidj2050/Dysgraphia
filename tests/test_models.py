import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dl_data import CANVAS_H, CANVAS_W, image_to_canvas, ink_to_canvas  # noqa: E402
from image_features import NoHandwritingError  # noqa: E402
from modeling import DysgraphiaRNN, canvas_to_tensor, make_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _page():
    img = np.full((500, 900), 255, np.uint8)
    for i, txt in enumerate(["Hello world again", "second line of text"]):
        cv2.putText(img, txt, (40, 150 + 150 * i), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.2, 0, 4)
    return img


def test_canvas_shape_and_range():
    canvas, feats = image_to_canvas(_page())
    assert canvas.shape == (CANVAS_H, CANVAS_W) and canvas.dtype == np.uint8
    assert canvas.max() > 0 and len(feats) == 21


def test_canvas_is_left_aligned_with_empty_right_padding():
    ink = np.zeros((100, 300), np.uint8)
    ink[30:70, 10:60] = 1
    canvas = ink_to_canvas(ink, 40.0)
    cols = np.where(canvas.max(axis=0) > 0)[0]
    assert cols.min() <= 3 and cols.max() < CANVAS_W // 2


def test_blank_image_rejected():
    with pytest.raises(NoHandwritingError):
        image_to_canvas(np.full((300, 600), 255, np.uint8))


@pytest.mark.parametrize("name", ["cnn", "rnn"])
def test_models_output_one_logit_per_image(name):
    model = make_model(name).eval()
    x = canvas_to_tensor(np.random.randint(0, 255, (3, CANVAS_H, CANVAS_W), dtype=np.uint8))
    with torch.no_grad():
        assert model(x).shape == (3,)


def test_rnn_ignores_padding_columns():
    """Same writing, different amount of empty padding on the right -> same output."""
    torch.manual_seed(0)
    rnn = DysgraphiaRNN().eval()
    base = torch.zeros(1, 1, CANVAS_H, CANVAS_W)
    base[..., 20:40, 4:100] = torch.rand(1, 1, 20, 96)
    shifted = base.clone()
    shifted[..., 200:] = 0.0
    with torch.no_grad():
        assert torch.allclose(rnn(base), rnn(shifted), atol=1e-5)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "models", "metadata.json")),
                    reason="models not trained yet (python src/train.py)")
def test_predict_reports_model_used():
    from predict import analyze_image
    r = analyze_image(_page())
    assert r["model_name"] in ("CNN", "RNN") and r["final"] in ("YES", "NO")
    assert {m["name"] for m in r["models"]} == {"CNN", "RNN"}
    assert sum(m["selected"] for m in r["models"]) == 1
    assert 0.0 <= r["probability"] <= 1.0


def test_many_lines_are_split_into_chunks_with_same_letter_size():
    """A 10-line page must not be squeezed onto one canvas (this caused missed cases)."""
    from dl_data import image_to_canvases
    img = np.full((1800, 900), 255, np.uint8)
    for i in range(10):
        cv2.putText(img, "Hello world again", (40, 120 + 160 * i),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.0, 0, 4)
    chunks, _ = image_to_canvases(img)
    single, _ = image_to_canvases(img[:300])
    assert len(chunks) >= 4
    h = lambda c: int((c.max(axis=1) > 0).sum())     # occupied rows
    assert h(chunks[0]) >= 0.6 * h(single[0])        # letters not shrunk to a page-fit size
