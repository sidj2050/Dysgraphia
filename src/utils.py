from pathlib import Path
import json
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def percentile_dict(values, q):
    return {k: float(np.percentile(v, q)) for k, v in values.items() if len(v)}

def safe_float(x):
    return float(x) if np.isfinite(x) else 0.0
