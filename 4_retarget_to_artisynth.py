#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retarget_to_artisynth.py

Kinematic RETARGETING of RT-MRI tongue motion onto the actual ArtiSynth tongue
surface mesh (tongue3d/geometry/tongue.obj): image->model registration,
midsagittal MOTION transfer (relative to a rest frame), TEMPORAL + SPATIAL
smoothing of the motion, then left-right SYMMETRIC Gaussian-RBF skinning that
decays away from the dorsum (base stays fixed). Pure-Python counterpart to the
ArtiSynth muscle-activation inverse.

Outputs (OUT_DIR): retargeted_tongue.npy (T,Nverts,3 mm), retarget_midsag.png,
retarget_frames3d.png, retarget_motion.gif (real-time @ FPS), retargeted_objs/*.obj

입력/출력: output/Subject{N}/ (MRI_SUBJECT로 선택)
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.interpolate import RBFInterpolator
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from mri_paths import MRI_OUT, MRI_FIT_DIR, TONGUE_OBJ, print_paths

OUT_DIR = MRI_OUT
OBJ     = TONGUE_OBJ
FPS      = 5.0          # actual frame rate (user-confirmed) -> time axis & gif speed
REST     = 0
NCTRL    = 13
RBF_LEN  = 18.0        # skinning length scale (mm); larger = smoother/stiffer
SPATIAL_WIN = 3        # smooth control displacement along the curve (pts; 1=off)
TEMPORAL_WIN = 9       # Savitzky-Golay window over frames (odd; 1=off) ~1.8s @5fps
TEMPORAL_POLY = 2
EXPORT_OBJ_FRAMES = [0, 26, 52, 78, 104]


def load_obj(path):
    V, F = [], []
    for L in open(path):
        t = L.split()
        if not t: continue
        if t[0] == "v": V.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == "f": F.append([int(p.split("/")[0]) - 1 for p in t[1:4]])
    V = np.array(V); F = np.array(F)
    V = V * 1000.0; V[:, 0] += 2.0
    return V, F


def affine_image_to_model(reg_csv):
    img, mod = [], []
    for r in csv.DictReader(open(reg_csv)):
        img.append([float(r["imageX"]), float(r["imageY"])])
        mod.append([float(r["modelX"]), float(r["modelZ"])])
    img = np.array(img); mod = np.array(mod)
    A, *_ = np.linalg.lstsq(np.column_stack([img, np.ones(len(img))]), mod, rcond=None)
    return lambda xy: np.column_stack([xy, np.ones(len(xy))]) @ A


def resample(curve, n):
    d = np.r_[0, np.cumsum(np.hypot(np.diff(curve[:,0]), np.diff(curve[:,1])))]
    if d[-1] == 0: return np.repeat(curve[:1], n, 0)
    u = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(u, d, curve[:,0]), np.interp(u, d, curve[:,1])])


def model_dorsal_curve(V, nb=NCTRL):
    x = V[:, 0]; z = V[:, 2]
    xq = np.linspace(x.min(), x.max(), nb)
    half = (x.max() - x.min()) / nb
    zc = np.full(nb, np.nan)
    for i, xi in enumerate(xq):
        sel = z[np.abs(x - xi) <= half]
        if len(sel): zc[i] = sel.max()
    ok = ~np.isnan(zc); zc = np.interp(xq, xq[ok], zc[ok])
    k = np.ones(3) / 3
    zs = np.convolve(zc, k, mode="same"); zs[0], zs[-1] = zc[0], zc[-1]
    return np.column_stack([xq, zs])


def main():
    print_paths()
    V, F = load_obj(OBJ)
    Vxz = V[:, [0, 2]]
    dorsal = model_dorsal_curve(V)

    tgt = np.load(os.path.join(OUT_DIR, "tongue_targets.npy"))
    T, N, _ = tgt.shape
    to_model = affine_image_to_model(os.path.join(MRI_FIT_DIR, "registration.csv"))

    mri = np.stack([resample(to_model(tgt[k, :, :2]), NCTRL) for k in range(T)], 0)  # (T,NCTRL,2)
    delta = mri - mri[REST]

    # ---- smoothing: spatial (along curve) then temporal (along frames) ----
    if SPATIAL_WIN > 1:
        delta = uniform_filter1d(delta, SPATIAL_WIN, axis=1, mode="nearest")
    if TEMPORAL_WIN > 1 and T >= TEMPORAL_WIN:
        delta = savgol_filter(delta, TEMPORAL_WIN, TEMPORAL_POLY, axis=0, mode="interp")
    delta = delta - delta[REST]                      # re-anchor: model at rest on frame REST

    deformed = np.zeros((T, len(V), 3))
    for k in range(T):
        rbf = RBFInterpolator(dorsal, delta[k], kernel="gaussian",
                              epsilon=1.0 / RBF_LEN, degree=-1, smoothing=1e-3)
        d_xz = rbf(Vxz)
        Vd = V.copy(); Vd[:, 0] += d_xz[:, 0]; Vd[:, 2] += d_xz[:, 1]
        deformed[k] = Vd
    np.save(os.path.join(OUT_DIR, "retargeted_tongue.npy"), deformed)
    print(f"[out] retargeted_tongue.npy  {deformed.shape}  ({T/FPS:.1f}s @ {FPS}fps)")
    disp = np.linalg.norm(deformed - deformed[REST], axis=2)
    # frame-to-frame jitter metric (smoothness)
    jit = np.linalg.norm(np.diff(deformed, axis=0), axis=2).mean()
    print(f"[stat] vertex max disp {disp.max():.1f}mm, mean per-frame max {disp.max(1).mean():.1f}mm, "
          f"frame-to-frame jitter {jit:.2f}mm")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    md = V[np.abs(V[:,1]) < 3.0]
    for ax, k in zip(axs, [0, T//2, T-1]):
        ax.scatter(md[:,0], md[:,2], s=6, c="0.8", label="model midsag (rest)")
        ax.plot(dorsal[:,0] + delta[k][:,0], dorsal[:,1] + delta[k][:,1],
                "b-o", ms=3, label="model dorsum retargeted")
        ax.plot(mri[k][:,0], mri[k][:,1], "r--", lw=1.5, label="MRI contour (model frame)")
        ax.set_aspect("equal"); ax.set_title(f"frame {k}  (t={k/FPS:.1f}s)")
        ax.set_xlabel("x ant-post (mm)"); ax.set_ylabel("z up (mm)")
        if k == 0: ax.legend(fontsize=7)
    fig.suptitle(f"Midsagittal retargeting (smoothed) @ {FPS} fps")
    fig.savefig(os.path.join(OUT_DIR, "retarget_midsag.png"), dpi=120, bbox_inches="tight")
    plt.close(fig); print("[out] retarget_midsag.png")

    P = deformed.reshape(-1, 3)
    xl=(P[:,0].min(),P[:,0].max()); yl=(P[:,1].min(),P[:,1].max()); zl=(P[:,2].min(),P[:,2].max())
    def draw(ax, k):
        Vd = deformed[k]
        ax.plot_trisurf(Vd[:,0], Vd[:,1], Vd[:,2], triangles=F,
                        cmap="viridis", alpha=0.9, linewidth=0.1, edgecolor="0.3")
        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
        ax.set_xlabel("x"); ax.set_ylabel("y lat"); ax.set_zlabel("z up")
        ax.view_init(elev=20, azim=-70); ax.set_title(f"t={k/FPS:.1f}s")
    fig = plt.figure(figsize=(15,5))
    for j,k in enumerate([0, T//2, T-1]):
        draw(fig.add_subplot(1,3,j+1, projection="3d"), k)
    fig.suptitle("ArtiSynth tongue mesh retargeted to MRI (smoothed)")
    fig.savefig(os.path.join(OUT_DIR,"retarget_frames3d.png"), dpi=120, bbox_inches="tight")
    plt.close(fig); print("[out] retarget_frames3d.png")

    step = 1
    frames = []
    for k in range(0, T, step):
        fig = plt.figure(figsize=(5,5)); ax = fig.add_subplot(111, projection="3d")
        draw(ax, k); fig.tight_layout(); fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames.append(buf); plt.close(fig)
    imageio.mimsave(os.path.join(OUT_DIR,"retarget_motion.gif"), frames, duration=step/FPS)  # real-time
    print(f"[out] retarget_motion.gif  ({len(frames)} frames, real-time @ {FPS}fps)")

    od = os.path.join(OUT_DIR, "retargeted_objs"); os.makedirs(od, exist_ok=True)
    for k in EXPORT_OBJ_FRAMES:
        if k >= T: continue
        with open(os.path.join(od, f"frame_{k:03d}.obj"), "w") as f:
            for v in deformed[k]: f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for tri in F: f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"[out] {od}/frame_*.obj  ({len(EXPORT_OBJ_FRAMES)} frames)")

if __name__ == "__main__":
    main()
