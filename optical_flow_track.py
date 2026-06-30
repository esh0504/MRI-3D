#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optical_flow_track.py

Turn 2D pixels into TRACKED points via optical flow: seed points on the rest
frame and advect them through per-frame flow so we get real trajectories
(solving the 2D correspondence problem), instead of re-detecting silhouette
extrema independently each frame.

skimage convention: optical_flow_ilk(ref, mov) -> (v,u) s.t. a point at (r,c) in
ref is at (r+v, c+u) in mov. Forward advection: p_{k+1}=p_k+flow_k(p_k).

Outputs (MRI_OUT):
  tongue_targets_flow.npy  / tongue_boundary_flow.npy / tongue_grid_flow.npy
  flow_track_trajectories.png, flow_vs_perframe.png (contour only)

Run:
  MRI_SUBJECT=Subject1 python3 optical_flow_track.py [--source contour|boundary|grid]
     [--mode chain|direct] [--n 25] [--rest 0] [--max-frames 0]
"""
import argparse
import glob
import os
import re

import numpy as np
import scipy.io as sio
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.registration import optical_flow_ilk

from tongue_contour import precise_contour, full_boundary_contour
from mri_paths import MRI_ROOT, MRI_OUT, print_paths

MM_PER_PX = 1.164
LBL_TONGUE = 4
FPS = float(os.environ.get("FPS", "5.0"))
CLIP_ROOT = os.environ.get("CLIP_ROOT", "1").lower() not in ("0", "false", "no")
CLIP_DROP_FRAC = float(os.environ.get("CLIP_DROP_FRAC", "1.0"))


def natkey(p):
    n = re.findall(r"\d+", os.path.basename(p)); return int(n[-1]) if n else 0


def find_dcm_dir():
    env = os.environ.get("DCM_DIR")
    if env and os.path.isdir(env):
        return env
    subj = os.path.basename(MRI_OUT.rstrip("/"))
    return os.path.join(os.path.dirname(os.path.dirname(MRI_ROOT.rstrip("/"))),
                        "MRI_SSFP_10fps", subj)


def load_cine(dcm_dir):
    import pydicom
    fs = sorted(glob.glob(os.path.join(dcm_dir, "image_*.dcm")), key=natkey)
    if not fs:
        raise SystemExit("no DICOM under %s (set DCM_DIR)" % dcm_dir)
    return np.stack([pydicom.dcmread(f).pixel_array.astype(np.float32) for f in fs], 0)


def load_masks(mask_dir):
    fs = sorted(glob.glob(os.path.join(mask_dir, "mask_*.mat")), key=natkey)
    out = []
    for f in fs:
        d = sio.loadmat(f)
        out.append(d.get("mask_frame", next(v for k, v in d.items() if not k.startswith("__"))))
    return np.stack(out, 0)


def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)


def seed_points(rest_mask, source, n):
    if source == "contour":
        return precise_contour(rest_mask, n, clip_root=CLIP_ROOT, clip_drop_frac=CLIP_DROP_FRAC)
    if source == "boundary":
        return full_boundary_contour(rest_mask, n)
    tg = (rest_mask == LBL_TONGUE)
    rr, cc = np.nonzero(tg)
    r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
    step = max(2, int(np.sqrt((r1 - r0) * (c1 - c0) / max(1, n))))
    pts = [(r, c) for r in range(r0, r1 + 1, step) for c in range(c0, c1 + 1, step) if tg[r, c]]
    return np.array(pts, dtype=float)


def sample_flow(field, rc):
    return map_coordinates(field, [rc[:, 0], rc[:, 1]], order=1, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["contour", "boundary", "grid"], default="contour")
    ap.add_argument("--mode", choices=["chain", "direct"], default="chain")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--rest", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    print_paths()

    dcm_dir = find_dcm_dir()
    cine = load_cine(dcm_dir)
    masks = load_masks(MRI_ROOT)
    T = min(len(cine), len(masks))
    if a.max_frames > 0:
        T = min(T, a.max_frames + 1)
    cine, masks = cine[:T], masks[:T]
    H = cine.shape[1]
    img = norm01(cine)
    print("[in] %s | %d frames %dx%d | source=%s mode=%s"
          % (dcm_dir, T, H, cine.shape[2], a.source, a.mode))

    seeds = seed_points(masks[a.rest], a.source, a.n)
    M = len(seeds)
    traj = np.zeros((T, M, 2)); traj[a.rest] = seeds

    if a.mode == "chain":
        cur = seeds.copy()
        for k in range(a.rest + 1, T):
            v, u = optical_flow_ilk(img[k - 1], img[k])
            cur = cur + np.column_stack([sample_flow(v, cur), sample_flow(u, cur)])
            traj[k] = cur
        cur = seeds.copy()
        for k in range(a.rest - 1, -1, -1):
            v, u = optical_flow_ilk(img[k + 1], img[k])
            cur = cur + np.column_stack([sample_flow(v, cur), sample_flow(u, cur)])
            traj[k] = cur
    else:
        for k in range(T):
            if k == a.rest:
                continue
            v, u = optical_flow_ilk(img[a.rest], img[k])
            traj[k] = seeds + np.column_stack([sample_flow(v, seeds), sample_flow(u, seeds)])

    X = traj[:, :, 1] * MM_PER_PX
    Y = (H - 1 - traj[:, :, 0]) * MM_PER_PX
    out = np.stack([X, Y, np.zeros_like(X)], axis=2)
    name = {"contour": "tongue_targets_flow.npy",
            "boundary": "tongue_boundary_flow.npy",
            "grid": "tongue_grid_flow.npy"}[a.source]
    np.save(os.path.join(MRI_OUT, name), out)
    print("[out] %s  %s  (flow-tracked, image mm)" % (name, out.shape))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img[a.rest], cmap="gray")
    for j in range(M):
        ax.plot(traj[:, j, 1], traj[:, j, 0], "-", lw=0.6, alpha=0.6)
    ax.scatter(seeds[:, 1], seeds[:, 0], s=14, c="cyan", zorder=3, label="rest seeds")
    ax.set_title("Flow-tracked trajectories (%s, %s)" % (a.source, a.mode))
    ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    p1 = os.path.join(MRI_OUT, "flow_track_trajectories.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("[out] %s" % p1)

    if a.source == "contour":
        kk = [a.rest, T // 2, T - 1]
        fig, axs = plt.subplots(1, len(kk), figsize=(5 * len(kk), 5))
        for ax, k in zip(np.atleast_1d(axs), kk):
            ax.imshow(img[k], cmap="gray")
            pf = precise_contour(masks[k], a.n, clip_root=CLIP_ROOT, clip_drop_frac=CLIP_DROP_FRAC)
            if pf is not None:
                ax.plot(pf[:, 1], pf[:, 0], "-", c="yellow", lw=1.5, label="per-frame contour")
            ax.plot(traj[k, :, 1], traj[k, :, 0], "-", c="red", lw=1.2, label="flow-tracked")
            ax.scatter(traj[k, :, 1], traj[k, :, 0], s=8, c="red")
            ax.set_title("frame %d (t=%.1fs)" % (k, k / FPS)); ax.legend(fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        p2 = os.path.join(MRI_OUT, "flow_vs_perframe.png")
        fig.savefig(p2, dpi=120, bbox_inches="tight"); plt.close(fig)
        print("[out] %s" % p2)


if __name__ == "__main__":
    main()
