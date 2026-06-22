#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mri_to_artisynth_targets.py

2D midsagittal RT-MRI tongue masks -> ArtiSynth motion targets.
Uses the PRECISE sub-pixel airway-facing surface extractor (tongue_contour.py).

Outputs: tongue_targets.npy (T,N,3), tongue_targets.txt (NumericInputProbe),
         qc_trajectories.png, resampled_markers.png
Image-mm space: x=col*MM_PER_PX, y=(H-1-row)*MM_PER_PX, z=0. Up=+y, anterior=small x.
Env overrides: MRI_ROOT, MRI_OUT.
"""
import os, re, glob
import numpy as np
import scipy.io as sio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tongue_contour import precise_contour

ROOT      = os.environ.get("MRI_ROOT", r"E:\Datasets\XAI\data\GT_Segmentations\Subject1")
OUT_DIR   = os.environ.get("MRI_OUT",  r"C:\Users\d11\Project\Tongue_Inverse")
VAR_NAME  = "mask_frame"
N_MARKERS = 25
MM_PER_PX = 1.164        # data-driven (from registration); was 1.0 placeholder
FPS       = 5.0          # actual frame rate (user-confirmed)


def natkey(p): n = re.findall(r"\d+", os.path.basename(p)); return int(n[-1]) if n else 0
def frames(): fs = glob.glob(os.path.join(ROOT, "mask_*.mat")); fs.sort(key=natkey); return fs
def load(p):
    d = sio.loadmat(p)
    return d[VAR_NAME] if VAR_NAME in d else next(v for k,v in d.items() if not k.startswith("__"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fs = frames(); assert fs, f"no masks under {ROOT}"
    H = load(fs[0]).shape[0]
    targets = []
    for i, fp in enumerate(fs):
        m = load(fp)
        c = precise_contour(m, N_MARKERS)              # (N,2) row,col tip->root
        if c is None:
            print(f"[warn] frame {i} unusable, skipped"); continue
        x = c[:,1]*MM_PER_PX; y = (H-1-c[:,0])*MM_PER_PX
        targets.append(np.column_stack([x, y, np.zeros_like(x)]))
    targets = np.stack(targets, 0)
    print(f"[info] {targets.shape[0]} frames, {N_MARKERS} markers")
    np.save(os.path.join(OUT_DIR,"tongue_targets.npy"), targets)
    with open(os.path.join(OUT_DIR,"tongue_targets.txt"),"w") as f:
        dt = 1.0/FPS
        for ti in range(targets.shape[0]):
            f.write(f"{ti*dt:.5f} " + " ".join(f"{v:.5f}" for v in targets[ti].reshape(-1)) + "\n")
    print(f"[out] tongue_targets.npy / .txt (vsize={N_MARKERS*3}, dt={1/FPS:.3f}s)")

    fig, ax = plt.subplots(figsize=(6,6))
    for mi in range(N_MARKERS): ax.plot(targets[:,mi,0], targets[:,mi,1], "-", lw=0.6, alpha=0.7)
    ax.scatter(targets[0,:,0], targets[0,:,1], c="k", s=12, zorder=3, label="frame0")
    ax.set_aspect("equal"); ax.set_xlabel("x (mm, ant->post)"); ax.set_ylabel("y (mm, up)")
    ax.set_title(f"Marker trajectories ({targets.shape[0]} frames, precise contour)"); ax.legend()
    fig.savefig(os.path.join(OUT_DIR,"qc_trajectories.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,6)); f0 = targets[0]
    ax.plot(f0[:,0], f0[:,1], "-o", ms=4)
    for mi in range(N_MARKERS): ax.annotate(str(mi),(f0[mi,0],f0[mi,1]),fontsize=6)
    ax.scatter(f0[0,0],f0[0,1],c="r",s=40,label="m0 tip"); ax.scatter(f0[-1,0],f0[-1,1],c="b",s=40,label=f"m{N_MARKERS-1} root")
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Resampled markers, frame 0 (precise)"); ax.legend()
    fig.savefig(os.path.join(OUT_DIR,"resampled_markers.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    print("[out] qc_trajectories.png, resampled_markers.png")
    disp = np.linalg.norm(targets - targets[0:1], axis=2)
    print(f"[stat] max disp {disp.max():.2f}mm, mean per-frame max {disp.max(1).mean():.2f}mm @ {MM_PER_PX}mm/px")

if __name__ == "__main__":
    main()
