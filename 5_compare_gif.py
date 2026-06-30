#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_compare_gif.py

Single side-by-side GIF, frame-synced @ actual frame rate:
  LEFT  = original RT-MRI segmentation (7-label colour) + tracked tongue markers
  RIGHT = retargeted ArtiSynth tongue mesh (from retargeted_tongue.npy)
Output: compare_mri_vs_retarget.gif

입력 GT: datasets/GT_Segmentations/Subject{N}/
출력: output/Subject{N}/ (MRI_SUBJECT로 선택)
"""
import os, re, glob
import numpy as np
import scipy.io as sio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio
from mri_paths import MRI_ROOT, MRI_OUT, TONGUE_OBJ, out, print_paths

ROOT    = MRI_ROOT
OUT_DIR = MRI_OUT
OBJ     = TONGUE_OBJ
# Match step 4: TARGETS_NPY picks which markers + retarget result to show.
TARGETS_NPY = os.environ.get("TARGETS_NPY", out(1, "tongue_targets.npy"))
_stem = os.path.splitext(os.path.basename(TARGETS_NPY))[0]
_after = _stem.split("tongue_targets")[-1].lstrip("_")
TAG = ("_" + _after) if _after else ""
FPS  = 5.0
STEP = 1                       # 1 = every frame (real-time); 2 = lighter file
CMAP = ListedColormap(["black","red","green","blue","orange","purple","skyblue"])

def natkey(p): n=re.findall(r"\d+",os.path.basename(p)); return int(n[-1]) if n else 0

def load_faces(path):
    F=[]
    for L in open(path):
        t=L.split()
        if t and t[0]=="f": F.append([int(p.split("/")[0])-1 for p in t[1:4]])
    return np.array(F)

def main():
    print_paths()
    fs = sorted(glob.glob(os.path.join(ROOT,"mask_*.mat")), key=natkey)
    masks = [sio.loadmat(f)["mask_frame"] for f in fs]
    T = len(masks)
    tgt = np.load(os.path.join(OUT_DIR, TARGETS_NPY))             # (T,N,3) image mm
    deformed = np.load(out(4, f"retargeted_tongue{TAG}.npy"))     # (T,Nv,3) mm
    print(f"[in] targets={os.path.basename(TARGETS_NPY)}, retarget=4_retargeted_tongue{TAG}.npy")
    F = load_faces(OBJ)
    H = masks[0].shape[0]

    # marker pixel coords: col = x, row = (H-1) - y
    mk_col = tgt[...,0]; mk_row = (H-1) - tgt[...,1]

    # crop the MRI to the oral region for clarity
    r0,r1,c0,c1 = 110, 220, 30, 170

    P = deformed.reshape(-1,3)
    xl=(P[:,0].min(),P[:,0].max()); yl=(P[:,1].min(),P[:,1].max()); zl=(P[:,2].min(),P[:,2].max())

    frames=[]
    for k in range(0,T,STEP):
        fig = plt.figure(figsize=(11,5))
        # left: MRI mask + tracked markers
        axL = fig.add_subplot(1,2,1)
        axL.imshow(masks[k][r0:r1,c0:c1], cmap=CMAP, vmin=0, vmax=6, interpolation="nearest")
        axL.plot(mk_col[k]-c0, mk_row[k]-r0, "-", c="white", lw=1.5)
        axL.scatter(mk_col[k]-c0, mk_row[k]-r0, c="cyan", s=10, zorder=3)
        axL.scatter(mk_col[k,0]-c0, mk_row[k,0]-r0, c="lime", s=45, marker="*", zorder=4)  # tip
        axL.set_title("Original RT-MRI (tongue=orange) + tracked surface")
        axL.set_xticks([]); axL.set_yticks([])
        # right: retargeted ArtiSynth tongue
        axR = fig.add_subplot(1,2,2, projection="3d")
        Vd = deformed[k]
        axR.plot_trisurf(Vd[:,0], Vd[:,1], Vd[:,2], triangles=F,
                         cmap="viridis", alpha=0.9, linewidth=0.1, edgecolor="0.3")
        axR.set_xlim(xl); axR.set_ylim(yl); axR.set_zlim(zl)
        axR.set_xlabel("x"); axR.set_ylabel("y lat"); axR.set_zlabel("z up")
        axR.view_init(elev=20, azim=-70)
        axR.set_title("Retargeted ArtiSynth tongue")
        fig.suptitle(f"MRI -> ArtiSynth retargeting    frame {k+1}/{T}   t = {k/FPS:.1f}s   @ {FPS} fps",
                     fontsize=12)
        fig.tight_layout(rect=[0,0,1,0.95]); fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1]+(4,))[...,:3]
        frames.append(buf); plt.close(fig)

    out_gif = out(5, f"compare_mri_vs_retarget{TAG}.gif")
    imageio.mimsave(out_gif, frames, duration=STEP/FPS)   # real-time @ FPS
    mb = os.path.getsize(out_gif)/1e6
    print(f"[out] {out_gif}  ({len(frames)} frames, {mb:.1f} MB, real-time @ {FPS}fps)")

if __name__ == "__main__":
    main()
