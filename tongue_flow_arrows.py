#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tongue_flow_arrows.py

Dense motion of the WHOLE tongue from the raw cine MRI (no per-point tracking):
optical flow between frames, drawn as ARROWS over the tongue region only.

  raw cine   datasets/MRI_SSFP_10fps/<subj>/image_*.dcm   (DICOM, intensity)
  tongue ROI datasets/GT_Segmentations/<subj>/mask_*.mat  (label 4)
  -> per consecutive frame pair: optical_flow_ilk -> (v_row, v_col) per pixel
  -> keep only vectors inside the tongue mask -> quiver on a downsampled grid
  -> GIF.

Modes:
  consecutive (default): frame k -> k+1   (instantaneous motion)
  rest:                  frame k -> REST   (cumulative displacement vs rest)

Output: MRI_OUT/tongue_flow_arrows.gif  (+ optional tongue_flow.npz with raw fields)

Run:
  MRI_SUBJECT=Subject1 python3 tongue_flow_arrows.py [--mode consecutive|rest]
      [--step 6] [--scale 3] [--max-frames 0] [--save-npz]
  --step  = quiver grid spacing (px);  --scale = arrow length multiplier
"""
import argparse
import glob
import os
import re

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage.registration import optical_flow_ilk

from mri_paths import MRI_ROOT, MRI_OUT, print_paths

DCM_DIR = os.path.join(os.path.dirname(MRI_ROOT.rstrip("/")),
                       "..", "MRI_SSFP_10fps", os.path.basename(MRI_ROOT.rstrip("/")))
FPS = float(os.environ.get("FPS", "5.0"))
LBL_TONGUE = 4


def natkey(p):
    n = re.findall(r"\d+", os.path.basename(p))
    return int(n[-1]) if n else 0


def find_dcm_dir():
    """Locate the raw cine dir for this subject (env DCM_DIR overrides)."""
    env = os.environ.get("DCM_DIR")
    if env and os.path.isdir(env):
        return env
    subj = os.path.basename(MRI_OUT.rstrip("/"))
    cand = os.path.join(os.path.dirname(os.path.dirname(MRI_ROOT.rstrip("/"))),
                        "MRI_SSFP_10fps", subj)
    return cand


def load_cine(dcm_dir):
    import pydicom
    fs = sorted(glob.glob(os.path.join(dcm_dir, "image_*.dcm")), key=natkey)
    if not fs:
        raise SystemExit("no DICOM under %s (set DCM_DIR)" % dcm_dir)
    imgs = [pydicom.dcmread(f).pixel_array.astype(np.float32) for f in fs]
    return np.stack(imgs, 0)


def load_masks(mask_dir):
    fs = sorted(glob.glob(os.path.join(mask_dir, "mask_*.mat")), key=natkey)
    out = []
    for f in fs:
        d = sio.loadmat(f)
        out.append(d.get("mask_frame", next(v for k, v in d.items()
                                            if not k.startswith("__"))))
    return np.stack(out, 0)


def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["consecutive", "rest"], default="consecutive")
    ap.add_argument("--rest", type=int, default=0, help="rest frame index (0-based)")
    ap.add_argument("--step", type=int, default=6, help="quiver grid spacing (px)")
    ap.add_argument("--scale", type=float, default=3.0, help="arrow length multiplier")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-npz", action="store_true")
    a = ap.parse_args()
    print_paths()

    dcm_dir = find_dcm_dir()
    print("[in] cine: %s" % dcm_dir)
    cine = load_cine(dcm_dir)
    masks = load_masks(MRI_ROOT)
    T = min(len(cine), len(masks))
    if a.max_frames > 0:
        T = min(T, a.max_frames + 1)
    cine, masks = cine[:T], masks[:T]
    print("[in] %d frames, %dx%d" % (T, cine.shape[1], cine.shape[2]))

    # tongue bbox across all frames (+margin) for a tight crop
    tg = (masks == LBL_TONGUE)
    rr, cc = np.nonzero(tg.any(0))
    r0, r1 = max(0, rr.min() - 10), min(cine.shape[1], rr.max() + 10)
    c0, c1 = max(0, cc.min() - 10), min(cine.shape[2], cc.max() + 10)

    img01 = norm01(cine)
    yy, xx = np.mgrid[r0:r1:a.step, c0:c1:a.step]   # quiver grid (rows, cols)

    # global magnitude scale (fixed across frames) for consistent arrow/colour
    sample = []
    rng = range(1, min(T, 12))
    for k in rng:
        ref = a.rest if a.mode == "rest" else k - 1
        v, u = optical_flow_ilk(img01[ref], img01[k])
        sample.append(np.hypot(v, u)[tg[k]].mean() if tg[k].any() else 0)
    vmax = max(1e-3, float(np.percentile(np.concatenate(
        [np.array(sample)]), 95)) * 4)

    frames_out, flow_store = [], []
    for k in range(1, T):
        ref = a.rest if a.mode == "rest" else k - 1
        v, u = optical_flow_ilk(img01[ref], img01[k])   # v=row disp, u=col disp
        m = tg[k]
        if a.save_npz:
            flow_store.append((np.where(m, v, 0).astype(np.float32),
                               np.where(m, u, 0).astype(np.float32)))

        # sample flow on the grid, keep only points inside the tongue
        gv = v[yy, xx]; gu = u[yy, xx]
        gm = m[yy, xx]
        Y, X = yy[gm], xx[gm]
        V, U = gv[gm], gu[gm]
        mag = np.hypot(U, V)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img01[k, r0:r1, c0:c1], cmap="gray",
                  extent=[c0, c1, r1, r0], vmin=0, vmax=1)
        if len(X):
            q = ax.quiver(X, Y, U, V, mag, cmap="turbo", clim=(0, vmax),
                          angles="xy", scale_units="xy", scale=1.0 / a.scale,
                          width=0.004, headwidth=4)
        ax.set_xlim(c0, c1); ax.set_ylim(r1, r0)
        ax.set_xticks([]); ax.set_yticks([])
        lbl = "vs rest %d" % a.rest if a.mode == "rest" else "%d->%d" % (ref, k)
        ax.set_title("Tongue optical-flow arrows  frame %s  (t=%.1fs)"
                     % (lbl, k / FPS), fontsize=10)
        fig.tight_layout(); fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames_out.append(buf); plt.close(fig)

    out = os.path.join(MRI_OUT, "tongue_flow_arrows.gif")
    imageio.mimsave(out, frames_out, duration=1.0 / FPS)
    print("[out] %s  (%d frames, %.1f MB)" % (out, len(frames_out),
                                              os.path.getsize(out) / 1e6))
    if a.save_npz and flow_store:
        vrow = np.stack([f[0] for f in flow_store]); ucol = np.stack([f[1] for f in flow_store])
        npz = os.path.join(MRI_OUT, "tongue_flow.npz")
        np.savez(npz, v_row=vrow, u_col=ucol, bbox=np.array([r0, r1, c0, c1]))
        print("[out] %s  (v_row,u_col %s)" % (npz, vrow.shape))


if __name__ == "__main__":
    main()
