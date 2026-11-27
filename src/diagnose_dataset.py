import os
import glob
import cv2

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

for folder in [
    "data/raw/potential_dysgraphia",
    "data/raw/low_potential_dysgraphia",
]:
    paths = []
    for ext in EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    paths = sorted(set(paths))
    valid = 0
    for p in paths:
        if cv2.imread(p) is not None:
            valid += 1
    print(f"{folder}: {len(paths)} files, {valid} readable images")
