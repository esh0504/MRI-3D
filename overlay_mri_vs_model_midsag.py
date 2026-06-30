#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overlay_mri_vs_model_midsag.py

Sanity bridge for "use the WHOLE contour": put the MRI full tongue boundary and
the MODEL midsagittal vertices in the SAME (model) frame and overlay them, so you
can see whether the full MRI outline matches the model's midsagittal silhouette
(the 37 y~=0 vertices you can actually map to).

Inputs (per MRI_SUBJECT):
  MRI_OUT/tongue_boundary.npy            (T,Nb,3) full MRI outline, image mm  [step 1]
  MRI_OUT/mri_fit/registration_m.csv     image[x,y] -> model[x,z] metres anchors [step 2]
  tongue_model/tongue_midsag_vertices.csv  model midsag verts (mm)  [color_midsag_vertices.py]
    (fallback: recompute from output/<subj>/retargeted_objs/frame_000.obj)

Output:
  MRI_OUT/overlay_mri_vs_model_midsag.png

Run:
  MRI_SUBJECT=Subject1 python3 overlay_mri_vs_model_midsag.py [--frame 1]
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mri_paths import MRI_OUT, MRI_FIT_DIR, out, print_paths

HERE = os.path.dirname(os.path.abspath(__file__))
X_OFFSET_MM = 2.0     # composite mm model = obj*1000 + 2mm in x (matches step 2/4)


def load_affine_m(path):
    """registration_m.csv -> affine A (3,2): image[x,y,1] -> model[x,z] metres."""
    img, mod = [], []
    for r in csv.DictReader(open(path)):
        img.append([float(r["imageX"]), float(r["imageY"])])
        mod.append([float(r["modelX"]), float(r["modelZ"])])
    img = np.array(img); mod = np.array(mod)
    A, *_ = np.linalg.lstsq(np.column_stack([img, np.ones(len(img))]), mod, rcond=None)
    return A


def load_midsag_model_mm():
    """Model midsag vertices in mm [x,z]. Prefer the CSV from color_midsag_vertices.py."""
    csvp = os.path.join(HERE, "tongue_model", "tongue_midsag_vertices.csv")
    if os.path.isfile(csvp):
        xz = []
        for r in csv.DictReader(open(csvp)):
            xz.append([float(r["x"]), float(r["z"])])
        return np.array(xz)
    # fallback: rest-frame obj, pick y~0
    objp = os.path.join(out(4, "retargeted_objs"), "frame_000.obj")
    V = []
    for line in open(objp):
        t = line.split()
        if t and t[0] == "v":
            V.append([float(t[1]), float(t[2]), float(t[3])])
    V = np.array(V)
    mid = np.abs(V[:, 1]) < 0.01 * abs(V[:, 1]).max()
    return V[mid][:, [0, 2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=1, help="MRI frame id (1-based)")
    a = ap.parse_args()
    print_paths()

    bnd = np.load(out(1, "tongue_boundary.npy"))   # (T,Nb,3) image mm
    A = load_affine_m(os.path.join(MRI_FIT_DIR, "registration_m.csv"))
    mids_mm = load_midsag_model_mm()                             # (M,2) model mm [x,z]

    fi = max(1, min(a.frame, len(bnd)))
    img_xy = bnd[fi - 1][:, :2]                                  # (Nb,2) image mm
    # image -> model metres -> mm (composite frame: *1000 already in metres; +2mm x)
    mod_m = np.column_stack([img_xy, np.ones(len(img_xy))]) @ A  # (Nb,2) metres [x,z]
    mri_mm = np.column_stack([mod_m[:, 0] * 1000.0 + X_OFFSET_MM, mod_m[:, 1] * 1000.0])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(np.r_[mri_mm[:, 0], mri_mm[0, 0]], np.r_[mri_mm[:, 1], mri_mm[0, 1]],
            "-", c="tab:blue", lw=1.5, label="MRI full contour (-> model frame)")
    ax.scatter(mri_mm[:, 0], mri_mm[:, 1], s=10, c="tab:blue")
    ax.scatter(mids_mm[:, 0], mids_mm[:, 1], s=45, c="crimson", marker="*",
               zorder=5, label="model midsag verts (y~0)")
    ax.set_aspect("equal"); ax.set_xlabel("x ant-post (mm)"); ax.set_ylabel("z up (mm)")
    ax.set_title("MRI full contour vs model midsagittal vertices  (frame %d)" % fi)
    ax.legend(fontsize=9)
    out_png = os.path.join(MRI_OUT, "overlay_mri_vs_model_midsag.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("[out] %s" % out_png)
    print("[stat] MRI contour %d pts, model midsag %d verts" % (len(mri_mm), len(mids_mm)))


if __name__ == "__main__":
    main()
