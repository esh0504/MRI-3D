#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_activations.py

Turn the ArtiSynth inverse output (subject1_computed_excitations.txt, produced by
JawFemMuscleTongueMriDemo) into per-FRAME and per-INTERVAL muscle-activation
tables + plots.

  computed_excitations.txt  is an ArtiSynth numeric probe file: header lines then
  rows of  "t  a0 a1 a2 ..."  where a_i = activation (0..1) of muscle exciter i.

Usage:
  python summarize_activations.py EXCITATIONS.txt [--fps 5] [--names GGP,GGA,...]
         [--segments 7]            # split into N equal time intervals
         [--intervals intervals.csv]  # columns: start,stop,label (seconds)

Outputs (next to the input):
  activations_per_frame.csv     frame,time,<muscle...>
  activations_per_interval.csv  label,start,stop,<muscle_mean...>,<muscle_peak...>
  activations_heatmap.png       muscles x time
  activations_peaks.png         peak activation per muscle
The exciter ORDER follows tongue.getMuscleExciters() (then jaw). If you don't pass
--names, columns are ex0..exN; the typical Buchaillard-tongue order is given below.
"""
import sys, os, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Typical bilateral exciter names for the FemMuscleTongueDemo tongue (edit to match
# your model's getMuscleExciters() order if needed):
DEFAULT_NAMES = ["GGP","GGM","GGA","STY","GH","MH","HG","VERT","TRANS","IL","SL"]


def parse_probe(path):
    """Read an ArtiSynth numeric probe text file -> (time[T], data[T,M])."""
    rows = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.replace(",", " ").split()
        try:
            vals = [float(t) for t in toks]
        except ValueError:
            continue                      # header / non-numeric line
        if len(vals) >= 2:
            rows.append(vals)
    if not rows:
        raise SystemExit("no numeric data rows found in " + path)
    width = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == width]   # keep the consistent data block
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1:]                  # time, activations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--names", default="")
    ap.add_argument("--segments", type=int, default=0)
    ap.add_argument("--intervals", default="")
    a = ap.parse_args()

    t, A = parse_probe(a.file)
    T, M = A.shape
    out = os.path.dirname(os.path.abspath(a.file))

    names = [s.strip() for s in a.names.split(",") if s.strip()]
    if len(names) != M:
        names = (DEFAULT_NAMES + [f"ex{i}" for i in range(M)])[:M] \
                if M <= len(DEFAULT_NAMES) else [f"ex{i}" for i in range(M)]

    # ---- per-frame table ----
    with open(os.path.join(out, "activations_per_frame.csv"), "w") as f:
        f.write("frame,time," + ",".join(names) + "\n")
        for i in range(T):
            f.write(f"{i+1},{t[i]:.4f}," + ",".join(f"{v:.5f}" for v in A[i]) + "\n")
    print(f"[out] activations_per_frame.csv  ({T} frames x {M} muscles)")

    # ---- intervals ----
    segs = []
    if a.intervals and os.path.exists(a.intervals):
        import csv
        for r in csv.DictReader(open(a.intervals)):
            segs.append((float(r["start"]), float(r["stop"]), r.get("label", "")))
    elif a.segments > 0:
        edges = np.linspace(t[0], t[-1], a.segments + 1)
        segs = [(edges[i], edges[i+1], f"seg{i+1}") for i in range(a.segments)]
    if segs:
        with open(os.path.join(out, "activations_per_interval.csv"), "w") as f:
            f.write("label,start,stop," + ",".join(n+"_mean" for n in names)
                    + "," + ",".join(n+"_peak" for n in names) + "\n")
            for s, e, lab in segs:
                sel = (t >= s) & (t <= e)
                if sel.sum() == 0: continue
                mean = A[sel].mean(0); peak = A[sel].max(0)
                f.write(f"{lab},{s:.3f},{e:.3f}," + ",".join(f"{v:.5f}" for v in mean)
                        + "," + ",".join(f"{v:.5f}" for v in peak) + "\n")
        print(f"[out] activations_per_interval.csv  ({len(segs)} intervals)")

    # ---- heatmap ----
    fig, ax = plt.subplots(figsize=(11, 0.4*M + 1.5))
    im = ax.imshow(A.T, aspect="auto", origin="lower", cmap="hot",
                   extent=[t[0], t[-1], -0.5, M-0.5], vmin=0, vmax=max(0.1, A.max()))
    ax.set_yticks(range(M)); ax.set_yticklabels(names)
    ax.set_xlabel("time (s)"); ax.set_title("Muscle activation over time")
    fig.colorbar(im, ax=ax, label="activation")
    fig.savefig(os.path.join(out, "activations_heatmap.png"), dpi=130, bbox_inches="tight")
    plt.close(fig); print("[out] activations_heatmap.png")

    # ---- peak per muscle ----
    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(A.max(0))[::-1]
    ax.bar([names[i] for i in order], A.max(0)[order], color="firebrick")
    ax.set_ylabel("peak activation"); ax.set_title("Peak activation per muscle")
    plt.xticks(rotation=45, ha="right")
    fig.savefig(os.path.join(out, "activations_peaks.png"), dpi=130, bbox_inches="tight")
    plt.close(fig); print("[out] activations_peaks.png")

if __name__ == "__main__":
    main()
