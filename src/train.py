import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_data import CANVAS_H, CANVAS_W, ROOT, U_OUT, load_cache      # noqa: E402
from image_features import FEATURE_NAMES                              # noqa: E402
from modeling import (MODEL_DESCRIPTIONS, MODEL_NAMES, canvas_to_tensor,  # noqa: E402
                      count_parameters, make_model)

warnings.filterwarnings("ignore")
SEED = 42
ARCHS = ["cnn", "rnn"]
EPOCHS, BATCH, LR, WEIGHT_DECAY, SMOOTH = 45, 16, 3e-3, 1e-2, 0.05


# ------------------------------------------------------------ augmentation
def augment(x):
    """Random small geometric jitter + stroke thickening on a batch (B,1,H,W)."""
    B, _, H, W = x.shape
    u = lambda lim: (torch.rand(B) * 2 - 1) * lim
    ang, sc, sx, shear = u(math.radians(3)), 1 + u(0.10), 1 + u(0.08), u(0.10)
    tx, ty = u(0.10), u(0.08)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0] = cos / (sc * sx)
    theta[:, 0, 1] = -sin * (H / W) / sc + shear
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * (W / H) / (sc * sx)
    theta[:, 1, 1] = cos / sc
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    thick = (torch.rand(B) < 0.3).view(B, 1, 1, 1)
    return torch.where(thick, F.max_pool2d(x, 3, 1, 1), x).clamp(0, 1)


# --------------------------------------------------------------- training
def train_one(name, X, y, imgs, tr_rows, seed, epochs=EPOCHS):
    """Train one network on the rows ``tr_rows``; returns (model, loss_history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(name)
    by_img = pd.Series(tr_rows).groupby(imgs[tr_rows]).apply(np.array).tolist()
    n = len(by_img)
    labels = np.array([y[r[0]] for r in by_img], dtype=np.float32)
    pos_weight = torch.tensor(float((labels == 0).sum() / max((labels == 1).sum(), 1)))
    steps = math.ceil(n / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * steps)
    hist = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for s in range(steps):
            b = order[s * BATCH:(s + 1) * BATCH]
            if len(b) < 2:
                continue
            rows = [by_img[i][rng.integers(len(by_img[i]))] for i in b]
            xb = augment(canvas_to_tensor(X[rows]))
            yb = torch.tensor(labels[b]) * (1 - SMOOTH) + 0.5 * SMOOTH
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss) * len(b)
        hist.append(total / n)
    return model, hist


@torch.no_grad()
def predict_proba(model, X, batch=256):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(torch.sigmoid(model(canvas_to_tensor(X[i:i + batch]))).numpy())
    return np.concatenate(out)


def aggregate(p, codes):
    """Combine the chunk (text-line) probabilities of each image: mean of logits."""
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = np.bincount(codes, lg) / np.bincount(codes)
    return 1 / (1 + np.exp(-m))


# --------------------------------------------------------------- metrics
TARGET_SENSITIVITY = 0.90


def best_threshold(y, p, target=TARGET_SENSITIVITY):
    """Screening operating point: the highest threshold that still catches at least
    ``target`` of the potential-dysgraphia samples (missing a real case is worse than
    a false alarm).  Falls back to the most sensitive threshold if the target is
    unreachable."""
    ts = np.arange(0.80, 0.0499, -0.01)
    sens = np.array([recall_score(y, (p >= t).astype(int)) for t in ts])
    ok = np.where(sens >= target)[0]
    return float(ts[ok[0]] if len(ok) else ts[-1])


def summarise(y, p, t):
    pred = (p >= t).astype(int)
    return dict(
        balanced_accuracy=balanced_accuracy_score(y, pred),
        sensitivity=recall_score(y, pred), specificity=recall_score(y, pred, pos_label=0),
        precision=precision_score(y, pred, zero_division=0),
        f1=f1_score(y, pred), roc_auc=roc_auc_score(y, p))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    print("DATASET")
    df, X = load_cache(rebuild=args.rebuild)
    if "error" in df.columns:
        bad = df["error"].notna().values
        print(f"  skipped {int(bad.sum())} unusable samples")
        df, X = df[~bad].reset_index(drop=True), X[~bad]
    y = df["label"].values.astype(int)
    imgs = df["img"].values + "|" + df["label"].astype(str).values
    key = (df["img"] + "|" + df["variant"].astype(str)).values
    codes, _ = pd.factorize(key)
    first = np.unique(codes, return_index=True)[1]       # first chunk-row of every image
    yk = y[first]
    clean_k = (df["variant"].values == 0)[first]
    photo_k = ~clean_k
    clean = (df["variant"] == 0).values
    print(f"  chunks: {len(df)} | samples: {len(first)} | images: {len(set(imgs))} | children: {df['child'].nunique()}"
          f" | canvas: {CANVAS_H}x{CANVAS_W}")
    for a in ARCHS:
        print(f"  {MODEL_NAMES[a]} parameters: {count_parameters(make_model(a)):,}")

    print(f"\nCROSS-VALIDATION (grouped by child, {args.repeats} x {args.folds}-fold)")
    oof = {a: np.zeros((args.repeats, len(df))) for a in ARCHS}
    nets = {a: [] for a in ARCHS}
    hist = {a: [] for a in ARCHS}
    t0 = time.time()
    for r in range(args.repeats):
        cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED + r)
        for k, (tr, te) in enumerate(cv.split(X, y, groups=df["child"].values)):
            for a in ARCHS:
                model, h = train_one(a, X, y, imgs, tr, seed=SEED + 100 * r + k, epochs=args.epochs)
                oof[a][r, te] = predict_proba(model, X[te])
                nets[a].append({k_: v.clone() for k_, v in model.state_dict().items()})
                hist[a].append(h)
                fold_auc = roc_auc_score(y[te], oof[a][r, te])
                print(f"  repeat {r + 1} fold {k + 1}/{args.folds} {MODEL_NAMES[a]:>3}: "
                      f"loss {h[-1]:.3f}  val AUC {fold_auc:.3f}  [{time.time() - t0:.0f}s]",
                      flush=True)

    rows, thresholds, per_model = [], {}, {}
    for a in ARCHS:
        p = aggregate(oof[a].mean(axis=0), codes)
        t = best_threshold(yk, p)
        ca, ph = summarise(yk[clean_k], p[clean_k], t), summarise(yk[photo_k], p[photo_k], t)
        thresholds[a] = t
        per_model[a] = dict(clean=ca, photo=ph, p=p)
        rows.append(dict(model=MODEL_NAMES[a], threshold=t, parameters=count_parameters(make_model(a)),
                         clean_sens=ca["sensitivity"], clean_spec=ca["specificity"],
                         clean_bal_acc=ca["balanced_accuracy"], clean_auc=ca["roc_auc"],
                         photo_sens=ph["sensitivity"], photo_spec=ph["specificity"],
                         photo_bal_acc=ph["balanced_accuracy"], photo_auc=ph["roc_auc"]))
    comp = pd.DataFrame(rows)
    comp["score"] = 0.5 * comp["clean_auc"] + 0.5 * comp["photo_auc"]
    print("\n" + comp.round(3).to_string(index=False, max_cols=20))
    best_label = comp.sort_values("score", ascending=False).iloc[0]["model"]
    best = next(a for a in ARCHS if MODEL_NAMES[a] == best_label)
    thr = thresholds[best]
    print(f"\nSelected model: {MODEL_NAMES[best]}  (threshold {thr:.2f})")

    # reference ranges = typical writers (low-potential class), clean originals
    ref_df = df[(df["label"] == 0) & clean & (df["chunk"] == 0)]
    reference = {c: {"p10": float(np.percentile(ref_df[c], 10)),
                     "p50": float(np.percentile(ref_df[c], 50)),
                     "p90": float(np.percentile(ref_df[c], 90))} for c in FEATURE_NAMES}

    models_dir, reports_dir = os.path.join(ROOT, "models"), os.path.join(ROOT, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    models_meta = {}
    for a in ARCHS:
        torch.save({"arch": a, "state_dicts": nets[a]}, os.path.join(models_dir, f"{a}.pt"))
        models_meta[a] = {
            "name": MODEL_NAMES[a], "description": MODEL_DESCRIPTIONS[a],
            "decision_threshold": thresholds[a],
            "parameters": count_parameters(make_model(a)), "fold_networks": len(nets[a]),
            "cross_validation": {"clean_images": per_model[a]["clean"],
                                 "photo_style_images": per_model[a]["photo"]}}
    meta = {
        "feature_cols": list(FEATURE_NAMES),
        "best_model": best, "best_model_name": MODEL_NAMES[best],
        "decision_threshold": thr, "models": models_meta,
        "reference_distributions": reference,
        "class_names": {"0": "low_potential_dysgraphia", "1": "potential_dysgraphia"},
        "input": {"canvas_height": CANVAS_H, "canvas_width": CANVAS_W, "letter_height_px": U_OUT},
        "cross_validation": {
            "protocol": f"{args.repeats}x repeated {args.folds}-fold, grouped by child "
                        "(no child in train and validation)",
            "operating_point": f"threshold = highest value with >= {TARGET_SENSITIVITY:.0%} "
                               "sensitivity in out-of-fold predictions",
            "clean_images": per_model[best]["clean"],
            "photo_style_images": per_model[best]["photo"]},
        "training_counts": {
            "potential_dysgraphia": int(((yk == 1) & clean_k).sum()),
            "low_potential_dysgraphia": int(((yk == 0) & clean_k).sum())},
        "warning": "Research/screening support only. Not a clinical diagnosis.",
    }
    with open(os.path.join(models_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    comp.round(4).to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)
    cms = {a: confusion_matrix(yk[clean_k], (per_model[a]["p"][clean_k] >= thresholds[a]).astype(int),
                               labels=[0, 1]) for a in ARCHS}
    pd.DataFrame(cms[best], index=["Actual NO", "Actual YES"],
                 columns=["Predicted NO", "Predicted YES"]).to_csv(
        os.path.join(reports_dir, "confusion_matrix.csv"))
    pd.DataFrame({f"{MODEL_NAMES[a]}_fold{i + 1}": h for a in ARCHS
                  for i, h in enumerate(hist[a])}).rename_axis("epoch").to_csv(
        os.path.join(reports_dir, "training_loss.csv"))
    pd.DataFrame({"img": [k.split("|")[0] for k in key[first]], "variant": df["variant"].values[first],
                  "label": yk, **{f"p_{a}": per_model[a]["p"] for a in ARCHS}}).to_csv(
        os.path.join(reports_dir, "oof_predictions.csv"), index=False)
    with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
        f.write(f"Selected model: {MODEL_NAMES[best]}\n"
                f"Threshold (from out-of-fold predictions): {thr:.2f}\n"
                f"Validation: {args.repeats}x repeated {args.folds}-fold CV grouped by child.\n\n")
        for a in ARCHS:
            f.write(f"=== {MODEL_NAMES[a]} (threshold {thresholds[a]:.2f}) ===\n")
            for tag, s in (("Clean originals", per_model[a]["clean"]),
                           ("Photo-style renderings", per_model[a]["photo"])):
                f.write(f"{tag}\n" + "\n".join(f"  {k:>18}: {v:.3f}" for k, v in s.items()) + "\n")
            f.write("Confusion matrix (clean originals, out-of-fold)\n" + str(cms[a]) + "\n\n")
    print("\nSaved models + reports.")


if __name__ == "__main__":
    main()
