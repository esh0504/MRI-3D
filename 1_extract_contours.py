#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_extract_contours.py

2D midsagittal RT-MRI tongue masks -> ArtiSynth motion targets.
Outputs: tongue_targets.npy (T,N,3) dorsal arc (spur-trimmed if CLIP_ROOT),
         tongue_boundary.npy (T,Nb,3) full closed outline,
         landmarks_auto.csv (tip/dorsum/root/floor per frame),
         tongue_targets.txt, qc_trajectories.png, resampled_markers.png,
         qc_boundary_landmarks.png
Image-mm space: x=col*MM_PER_PX, y=(H-1-row)*MM_PER_PX, z=0.
"""
import os, re, glob
import numpy as np
import scipy.io as sio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tongue_contour import precise_contour, full_boundary_contour, anatomical_landmarks
from mri_paths import MRI_ROOT, MRI_OUT, out, print_paths

ROOT      = MRI_ROOT
OUT_DIR   = MRI_OUT
VAR_NAME  = "mask_frame"
N_MARKERS = 25
N_BOUNDARY = int(os.environ.get("N_BOUNDARY", "80"))   # full closed-outline points
LANDMARK_NAMES = ["tip", "dorsum", "root", "floor"]
# CLIP_ROOT: trim the posterior spur from the dorsal arc so the 2D tongue extent
# matches the 3D model (no pharyngeal/curl-back limb). CLIP_DROP_FRAC ~1.0 ends at
# the root shoulder; 0 = x-reversal only.
CLIP_ROOT = os.environ.get("CLIP_ROOT", "1").lower() not in ("0", "false", "no")
CLIP_DROP_FRAC = float(os.environ.get("CLIP_DROP_FRAC", "1.0"))
MM_PER_PX = 1.164
FPS       = 5.0


def natkey(p):
    n = re.findall(r"\d+", os.path.basename(p)); return int(n[-1]) if n else 0

def frames():
    fs = glob.glob(os.path.join(ROOT, "mask_*.mat")); fs.sort(key=natkey); return fs

def load(p):
    d = sio.loadmat(p)
    return d[VAR_NAME] if VAR_NAME in d else next(v for k, v in d.items() if not k.startswith("__"))


def main():
    print_paths()
    os.makedirs(OUT_DIR, exist_ok=True)
    fs = frames(); assert fs, f"no masks under {ROOT}"
    H = load(fs[0]).shape[0]
    targets, boundary, lm_rows = [], [], []
    for i, fp in enumerate(fs):
        fi = i + 1
        m = load(fp)
        c = precise_contour(m, N_MARKERS, clip_root=CLIP_ROOT, clip_drop_frac=CLIP_DROP_FRAC)
        b = full_boundary_contour(m, N_BOUNDARY)
        if c is None or b is None:
            print(f"[warn] frame {fi} unusable, skipped"); continue
        x = c[:, 1] * MM_PER_PX; y = (H - 1 - c[:, 0]) * MM_PER_PX
        targets.append(np.column_stack([x, y, np.zeros_like(x)]))
        bx = b[:, 1] * MM_PER_PX; by = (H - 1 - b[:, 0]) * MM_PER_PX
        boundary.append(np.column_stack([bx, by, np.zeros_like(bx)]))
        lm = anatomical_landmarks(m)
        if lm is not None:
            for name in LANDMARK_NAMES:
                if name in lm:
                    r, cc = lm[name]
                    lm_rows.append((fi, name, cc * MM_PER_PX, (H - 1 - r) * MM_PER_PX))
    targets = np.stack(targets, 0); boundary = np.stack(boundary, 0)
    print(f"[info] {targets.shape[0]} frames | dorsal {N_MARKERS} pts (clip={CLIP_ROOT}) | boundary {N_BOUNDARY} pts")
    np.save(out(1, "tongue_targets.npy"), targets)
    np.save(out(1, "tongue_boundary.npy"), boundary)
    with open(out(1, "landmarks_auto.csv"), "w") as f:
        f.write("frame,label,x,y\n")
        for fi, name, x, y in lm_rows:
            f.write(f"{fi},{name},{x:.3f},{y:.3f}\n")
    print(f"[out] 1_tongue_boundary.npy {boundary.shape}, 1_landmarks_auto.csv ({len(lm_rows)} rows)")

    with open(out(1, "tongue_targets.txt"), "w") as f:
        dt = 1.0 / FPS
        for ti in range(targets.shape[0]):
            f.write(f"{ti*dt:.5f} " + " ".join(f"{v:.5f}" for v in targets[ti].reshape(-1)) + "\n")
    print(f"[out] 1_tongue_targets.npy / .txt")

    fig, ax = plt.subplots(figsize=(6, 6))
    for mi in range(targets.shape[1]):
        ax.plot(targets[:, mi, 0], targets[:, mi, 1], "-", lw=0.6, alpha=0.7)
    ax.scatter(targets[0, :, 0], targets[0, :, 1], c="k", s=12, zorder=3, label="frame0")
    ax.set_aspect("equal"); ax.set_xlabel("x (mm, ant->post)"); ax.set_ylabel("y (mm, up)")
    ax.set_title(f"Marker trajectories ({targets.shape[0]} frames)"); ax.legend()
    fig.savefig(out(1, "qc_trajectories.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6)); f0 = targets[0]
    ax.plot(f0[:, 0], f0[:, 1], "-o", ms=4)
    for mi in range(targets.shape[1]):
        ax.annotate(str(mi), (f0[mi, 0], f0[mi, 1]), fontsize=6)
    ax.scatter(f0[0, 0], f0[0, 1], c="r", s=40, label="m0 tip")
    ax.scatter(f0[-1, 0], f0[-1, 1], c="b", s=40, label="root end")
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Resampled dorsal markers, frame 0 (spur-trimmed)"); ax.legend()
    fig.savefig(out(1, "resampled_markers.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    print("[out] 1_qc_trajectories.png, 1_resampled_markers.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    b0 = boundary[0]
    ax.plot(np.r_[b0[:, 0], b0[0, 0]], np.r_[b0[:, 1], b0[0, 1]], "-", c="0.5", lw=1, label="full boundary")
    ax.scatter(b0[:, 0], b0[:, 1], s=8, c="tab:blue")
    ax.plot(targets[0][:, 0], targets[0][:, 1], "-", c="tab:green", lw=2, label="dorsal arc (targets)")
    first_fi = lm_rows[0][0] if lm_rows else None
    lm0 = {name: (cc, yy) for fi, name, cc, yy in lm_rows if fi == first_fi}
    for name, (lx, ly) in lm0.items():
        ax.scatter([lx], [ly], s=70, marker="*", zorder=5)
        ax.annotate(name, (lx, ly), fontsize=9, weight="bold")
    ax.set_aspect("equal"); ax.set_xlabel("x (mm, ant->post)"); ax.set_ylabel("y (mm, up)")
    ax.set_title("Full boundary + anatomical landmarks (frame 0)"); ax.legend(fontsize=8)
    fig.savefig(out(1, "qc_boundary_landmarks.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    print("[out] 1_qc_boundary_landmarks.png")

    disp = np.linalg.norm(targets - targets[0:1], axis=2)
    print(f"[stat] max disp {disp.max():.2f}mm, mean per-frame max {disp.max(1).mean():.2f}mm @ {MM_PER_PX}mm/px")


if __name__ == "__main__":
    main()
