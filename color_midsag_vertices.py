#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
color_midsag_vertices.py

Take an ArtiSynth tongue mesh (.obj) and mark the MIDSAGITTAL vertices (those on
the model symmetry plane y~=0) in a different colour. These are exactly the
vertices that correspond to the 2D midsagittal RT-MRI contour, so they are the
ones you can map / register against the MRI.

Outputs (next to the input, or OUT dir):
  tongue_midsag_colored.obj   per-vertex colours: midsag = red, others = grey
                              (uses the "v x y z r g b" extension; opens in MeshLab)
  tongue_midsag_vertices.csv  index(0-based),objIndex(1-based),x,y,z of midsag verts
  tongue_midsag_preview.png   3D scatter highlighting the midsag vertices

Midsag detection: |y| < THRESH. THRESH defaults to 1% of the half-width in y
(unit-independent: works whether the obj is in mm or metres). Override with
absolute --thresh or env MIDSAG_THRESH.

Usage:
  python3 color_midsag_vertices.py [tongue.obj] [--thresh 0.5] [--out DIR]
  TONGUE_OBJ=/path/tongue.obj python3 color_midsag_vertices.py
"""
import argparse
import os
import sys

import numpy as np


def load_obj(path):
    """Return (V (N,3) float, faces list of vertex-index lists, raw_face_lines)."""
    V, faces = [], []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == "f":
            faces.append([p.split("/")[0] for p in t[1:]])   # keep as strings (1-based)
    return np.array(V, dtype=float), faces


def write_colored_obj(path, V, faces, is_mid, c_mid=(0.85, 0.10, 0.10),
                      c_other=(0.72, 0.72, 0.72)):
    """Write OBJ with per-vertex colours via the 'v x y z r g b' extension."""
    with open(path, "w") as f:
        f.write("# midsagittal vertices coloured red; written by color_midsag_vertices.py\n")
        for i, (x, y, z) in enumerate(V):
            r, g, b = c_mid if is_mid[i] else c_other
            f.write("v %.6f %.6f %.6f %.4f %.4f %.4f\n" % (x, y, z, r, g, b))
        for fc in faces:
            f.write("f " + " ".join(fc) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj", nargs="?",
                    default=os.environ.get("TONGUE_OBJ", ""),
                    help="input tongue .obj (default: $TONGUE_OBJ)")
    ap.add_argument("--thresh", type=float,
                    default=float(os.environ.get("MIDSAG_THRESH", "0")),
                    help="absolute |y| threshold; 0 => auto (1%% of y half-width)")
    ap.add_argument("--frac", type=float, default=0.01,
                    help="auto threshold = frac * max|y| (default 0.01)")
    ap.add_argument("--out", default="",
                    help="output dir (default: same dir as the obj)")
    a = ap.parse_args()

    if not a.obj or not os.path.isfile(a.obj):
        sys.exit("input obj not found: %r\n  pass a path or set TONGUE_OBJ." % a.obj)

    V, faces = load_obj(a.obj)
    if len(V) == 0:
        sys.exit("no vertices parsed from " + a.obj)
    y = V[:, 1]
    thr = a.thresh if a.thresh > 0 else a.frac * float(np.abs(y).max())
    is_mid = np.abs(y) < thr
    n_mid = int(is_mid.sum())
    print("[in]  %s : %d verts, %d faces" % (a.obj, len(V), len(faces)))
    print("[mid] |y| < %.4g  => %d midsagittal vertices (%.1f%%)"
          % (thr, n_mid, 100.0 * n_mid / len(V)))
    if n_mid == 0:
        sys.exit("no midsag vertices found; raise --thresh")

    out = a.out or os.path.dirname(os.path.abspath(a.obj))
    os.makedirs(out, exist_ok=True)
    obj_out = os.path.join(out, "tongue_midsag_colored.obj")
    csv_out = os.path.join(out, "tongue_midsag_vertices.csv")
    png_out = os.path.join(out, "tongue_midsag_preview.png")

    write_colored_obj(obj_out, V, faces, is_mid)
    print("[out] %s" % obj_out)

    idx = np.where(is_mid)[0]
    with open(csv_out, "w") as f:
        f.write("index0,objIndex1,x,y,z\n")
        for i in idx:
            f.write("%d,%d,%.6f,%.6f,%.6f\n" % (i, i + 1, V[i, 0], V[i, 1], V[i, 2]))
    print("[out] %s  (%d midsag vertex indices)" % (csv_out, n_mid))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(11, 5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(V[~is_mid, 0], V[~is_mid, 1], V[~is_mid, 2], s=4,
                   c="0.72", alpha=0.5, label="other")
        ax.scatter(V[is_mid, 0], V[is_mid, 1], V[is_mid, 2], s=22,
                   c="crimson", label="midsag (y~0)")
        ax.set_xlabel("x"); ax.set_ylabel("y lat"); ax.set_zlabel("z up")
        ax.view_init(elev=18, azim=-70); ax.set_title("3D"); ax.legend(fontsize=8)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(V[~is_mid, 0], V[~is_mid, 2], s=5, c="0.8")
        ax2.scatter(V[is_mid, 0], V[is_mid, 2], s=24, c="crimson")
        for i in idx:
            ax2.annotate(str(i + 1), (V[i, 0], V[i, 2]), fontsize=5)
        ax2.set_aspect("equal"); ax2.set_xlabel("x ant-post"); ax2.set_ylabel("z up")
        ax2.set_title("midsagittal (x-z), obj-index labels")
        fig.suptitle("Tongue midsagittal vertices (%d of %d)" % (n_mid, len(V)))
        fig.savefig(png_out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print("[out] %s" % png_out)
    except Exception as e:
        print("[note] preview skipped: %s" % e)


if __name__ == "__main__":
    main()
