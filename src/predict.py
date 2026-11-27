import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_data import image_to_canvases                                   # noqa: E402
from image_features import (FEATURE_NAMES, NoHandwritingError,        # noqa: E402,F401
                            feature_interpretations)
from modeling import (MODEL_DESCRIPTIONS, MODEL_NAMES, canvas_to_tensor,  # noqa: E402
                      make_model)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CACHE = {}


def load_metadata():
    with open(os.path.join(PROJECT_ROOT, "models", "metadata.json"), encoding="utf-8") as f:
        return json.load(f)


def active_model_info():
    """Which network the app classifies with (reads metadata only - no torch weights)."""
    meta = load_metadata()
    key = meta["best_model"]
    return {"key": key, "name": MODEL_NAMES[key], "description": MODEL_DESCRIPTIONS[key]}


def load_artifacts():
    """Metadata + the fold-ensemble of every trained network, loaded once."""
    if "art" not in _CACHE:
        meta = load_metadata()
        nets = {}
        for key in meta["models"]:
            path = os.path.join(PROJECT_ROOT, "models", f"{key}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found. Train the networks first: python src/train.py")
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            ensemble = []
            for state in ckpt["state_dicts"]:
                net = make_model(key)
                net.load_state_dict(state)
                ensemble.append(net.eval())
            nets[key] = ensemble
        _CACHE["art"] = (nets, meta)
    return _CACHE["art"]


@torch.no_grad()
def _probability(ensemble, canvases):
    """Probability for one image: every text-line chunk is scored by every fold
    network, chunk logits are averaged (so page size / number of lines does not matter)."""
    x = canvas_to_tensor(np.stack(canvases))
    logits = np.mean([net(x).numpy() for net in ensemble], axis=0)      # per chunk
    return float(1.0 / (1.0 + np.exp(-logits.mean())))


def analyze_image(image):
    """image: path, bytes, PIL image or BGR ndarray. Raises NoHandwritingError."""
    nets, meta = load_artifacts()
    canvases, features = image_to_canvases(image)

    # every trained network scores the image; the selected one makes the decision
    best = meta["best_model"]
    scores = []
    for key, ensemble in nets.items():
        p = _probability(ensemble, canvases)
        t = float(meta["models"][key]["decision_threshold"])
        scores.append({"key": key, "name": MODEL_NAMES[key], "probability": p,
                       "threshold": t, "positive": p >= t, "selected": key == best})
    scores.sort(key=lambda s: not s["selected"])
    chosen = scores[0]
    probability, threshold, positive = chosen["probability"], chosen["threshold"], chosen["positive"]

    findings = feature_interpretations(features, meta.get("reference_distributions", {}))
    unusual = [x for x in findings if x["status"] == "Unusual"]
    if positive:
        explanation = ("The handwriting shows a combined pattern that resembles the "
                       "potential-dysgraphia samples the network learned from.")
    else:
        explanation = ("The handwriting does not show a strong enough combined pattern "
                       "to resemble the potential-dysgraphia samples the network learned from.")
    return {
        "final": "YES" if positive else "NO",
        "probability": probability, "threshold": threshold,
        "borderline": abs(probability - threshold) < 0.08,
        "explanation": explanation,
        "unusual_count": len(unusual), "total": len(findings),
        "findings": findings, "features": features,
        "model": best, "model_name": MODEL_NAMES[best],
        "model_description": MODEL_DESCRIPTIONS[best],
        "models": scores, "chunks": len(canvases),
    }
