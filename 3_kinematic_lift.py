#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kinematic_lift.py

KINEMATIC baseline (handoff doc step 5a): no dynamics, no FEM. Lift the 2D
midsagittal tongue-surface curve to a 3D surface using the left-right SYMMETRY
assumption, to get an immediate 3D-shape / motion sanity check.

This is NOT the FEM-based lift -- the physically correct out-of-plane shape comes
from the ArtiSynth inverse run (see export_mri_fit.py). Here the lateral profile
is a parametric symmetric dome, purely for visualization / correspondence checking.

Input : tongue_targets.npy  (T,N,3)  midsagittal markers tip->root, x=col y=up z=0
Output: tongue_lift_3d.npy  (T, N, Nz, 3) lifted surface node trajectories (mm)
        lift_frames3d.png    3 representative frames, 3D
        lift_motion.gif      surface deforming over the sequence (fixed view)

입력/출력: output/Subject{N}/ (MRI_SUBJECT로 선택)
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from mri_paths import MRI_OUT, print_paths

OUT_DIR  = MRI_OUT
MM_PER_PX = 1.164      # data-driven, from registration (model tongue size / image span)
NZ        = 15         # lateral samples per side-to-side span
HALF_W    = 30.0       # max lateral half-width (mm); model tongue spans +-34mm
EDGE_DROP = 9.0        # how far the lateral edges fall below the midline crest (mm)
WIDTH_END = 0.35       # relative width at tip & root vs mid-body


def width_profile(s):
    """Half-width along normalized arclength s in [0,1]: narrow at tip/root,
    widest mid-body."""
    bump = np.sin(np.pi * s) ** 0.6
    return HALF_W * (WIDTH_END + (1 - WIDTH_END) * bump)


def lift_frame(curve_mm):
    """curve_mm: (N,2) midline (x,y) mm tip->root -> (N,Nz,3) symmetric dome."""
    N = len(curve_mm)
    s = np.linspace(0, 1, N)
    W = width_profile(s)                          # (N,)
    zt = np.linspace(-1, 1, NZ)                    # normalized lateral
    surf = np.zeros((N, NZ, 3))
    for i in range(N):
        x, y = curve_mm[i]
        z = W[i] * zt                             # lateral position
        # elliptical coronal dome: edges drop EDGE_DROP below crest
        drop = EDGE_DROP * (1 - np.sqrt(np.clip(1 - zt**2, 0, 1)))
        surf[i, :, 0] = x
        surf[i, :, 1] = y - drop
        surf[i, :, 2] = z
    return surf


def main():
    print_paths()
    t = np.load(os.path.join(OUT_DIR, "tongue_targets.npy"))   # (T,N,3) px
    T, N, _ = t.shape
    xy = t[..., :2] * MM_PER_PX                                 # px->mm
    lifted = np.stack([lift_frame(xy[k]) for k in range(T)], 0) # (T,N,NZ,3)
    np.save(os.path.join(OUT_DIR, "tongue_lift_3d.npy"), lifted)
    print(f"[out] tongue_lift_3d.npy  shape {lifted.shape}")

    # consistent axis limits
    P = lifted.reshape(-1, 3)
    xl = (P[:,0].min(), P[:,0].max()); yl = (P[:,1].min(), P[:,1].max())
    zl = (P[:,2].min(), P[:,2].max())

    def draw(ax, k):
        S = lifted[k]
        ax.plot_surface(S[...,0], S[...,2], S[...,1], cmap="viridis",
                        rstride=1, cstride=1, alpha=0.9, linewidth=0)
        ax.plot(xy[k][:,0], np.zeros(N), xy[k][:,1], "r-", lw=2)   # midline
        ax.set_xlim(xl); ax.set_ylim(zl); ax.set_zlim(yl)
        ax.set_xlabel("x ant-post"); ax.set_ylabel("z lateral"); ax.set_zlabel("y up")
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f"frame {k}")

    # 3 representative frames
    fig = plt.figure(figsize=(15,5))
    for j,k in enumerate([0, T//2, T-1]):
        ax = fig.add_subplot(1,3,j+1, projection="3d"); draw(ax, k)
    fig.suptitle("Kinematic symmetric 3D lift (midline red)")
    fig.savefig(os.path.join(OUT_DIR,"lift_frames3d.png"), dpi=120, bbox_inches="tight")
    plt.close(fig); print("[out] lift_frames3d.png")

    # animation GIF (every other frame to keep size down)
    frames = []
    for k in range(0, T, 2):
        fig = plt.figure(figsize=(5,5)); ax = fig.add_subplot(111, projection="3d")
        draw(ax, k); fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames.append(buf); plt.close(fig)
    imageio.mimsave(os.path.join(OUT_DIR,"lift_motion.gif"), frames, duration=0.08)
    print("[out] lift_motion.gif")

    disp = np.linalg.norm(lifted - lifted[0:1], axis=3)
    print(f"[stat] lift node max disp {disp.max():.1f}mm, mean per-frame max {disp.max((1,2)).mean():.1f}mm")

if __name__ == "__main__":
    main()
