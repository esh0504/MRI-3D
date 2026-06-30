#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9_compare_forward.py

Round-trip 2·3단계: forward(8번)로 뽑은 3D Tongue Motion을
  (2) RETARGET 메쉬(4번)와 정량 비교  → 프레임별 표면 오차 그래프/CSV
  (3) MRI 영상 + Retarget + Forward 를 한 화면에 시각 비교 GIF

핵심 변경(이전 버전 대비):
  이전에는 forward 를 inverse(6번)의 '입력 타깃'(goals.npz)과 비교했음. 그건
  inverse↔forward '왕복 닫힘(closure)'만 증명할 뿐, 예측한 근육값이 *retarget
  형상*을 재현하는지는 증명하지 못한다(goals 는 inverse 가 자기가 맞추려던 값이라
  순환 논리). 그래서 이제 forward 표면을 4번이 만든 retargeted_tongue.npy 와
  직접 비교한다. "근육값이 잘 예측되었나" = "forward 가 retarget 과 닮았나" 를
  바로 측정한다.

좌표/메쉬 주의:
  - retarget 메쉬: tongue.obj 를 mm 로(×1000) 올리고 x+2mm 이동해 변형(4번 load_obj).
  - forward 메쉬: 같은 ArtiSynth 혀의 FEM 표면을 metres 로 출력(8번).
  두 메쉬는 같은 혀지만 *정점 수가 다르다*(retarget 433 vs FEM 표면 370). 그래서
  정점 1:1 이 아니라 표면-대-표면 최근접 거리(Chamfer)로 잰다. forward 를
  ×1000 + x(+2mm) 로 retarget 좌표계에 맞춘 뒤 비교한다.
  (테셀레이션이 달라 rest 에서도 ~수 mm 베이스라인이 남는다 — 절대값보다 '시간에
   따른 변화/상대 비교'로 해석할 것.)

환경변수: MRI_SUBJECT, MRI_ROOT, MRI_OUT (공통), STEP(gif 프레임 간격), FPS

실행:
  python3 9_compare_forward.py
"""
import os
import re
import glob

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio
import scipy.io as sio
from scipy.spatial import cKDTree

from mri_paths import MRI_ROOT, MRI_OUT, TONGUE_OBJ, out, print_paths

FPS = float(os.environ.get("FPS", "5.0"))
STEP = int(os.environ.get("STEP", "1"))
CMAP = ListedColormap(["black", "red", "green", "blue", "orange", "purple", "skyblue"])
M2MM = 1000.0
# 4_retarget_to_artisynth.py load_obj()의 정렬과 동일하게 forward 를 retarget 좌표계로 맞춘다.
X_OFFSET_MM = float(os.environ.get("X_OFFSET_MM", "2.0"))


def natkey(p):
    n = re.findall(r"\d+", os.path.basename(p))
    return int(n[-1]) if n else 0


def load_faces(path):
    F = []
    for L in open(path):
        t = L.split()
        if t and t[0] == "f":
            F.append([int(p.split("/")[0]) - 1 for p in t[1:4]])
    return np.array(F)


def _forward_to_retarget_frame(fwd_v_m):
    """forward 표면 정점(metres, 모델좌표) → retarget 좌표계(mm, x+2mm)."""
    v = fwd_v_m * M2MM
    v[..., 0] += X_OFFSET_MM
    return v


# --------------------- (2) 정량 오차: forward vs retarget --------------------
def compute_surface_error():
    """forward 표면 ↔ retarget 표면 의 프레임별 최근접(표면-대-표면) 거리.

    반환: dict(frame_ids, mean_mm, p95_mm, max_mm, chamfer_mm,
               per_vertex[프레임별 forward 정점 오차 리스트]) 또는 None.
    """
    fwd_v_path = out(8, "forward_surface_verts.npy")
    rpath = out(4, "retargeted_tongue.npy")
    if not (os.path.isfile(fwd_v_path) and os.path.isfile(rpath)):
        print("[warn] 8_forward_surface_verts 또는 4_retargeted_tongue 없음 → 정량 오차 건너뜀")
        return None

    fwd = _forward_to_retarget_frame(np.load(fwd_v_path))     # (T,Nf,3) mm
    ret = np.load(rpath)                                       # (Tr,Nr,3) mm
    fframes = np.load(out(8, "forward_frame_ids.npy"))
    okp = out(8, "forward_ok.npy")
    ok = np.load(okp) if os.path.isfile(okp) else np.ones(len(fframes), bool)

    rows, mean_mm, p95_mm, max_mm, cham_mm, per_vertex = [], [], [], [], [], []
    for k, fr in enumerate(fframes):
        idx = int(fr) - 1                       # retarget 은 0-based, frame id 1 = rest
        if idx < 0 or idx >= len(ret):
            continue
        R, Fm = ret[idx], fwd[k]
        d_f2r = cKDTree(R).query(Fm)[0]         # forward 정점 -> 가장 가까운 retarget
        d_r2f = cKDTree(Fm).query(R)[0]         # retarget 정점 -> 가장 가까운 forward
        rows.append(int(fr))
        mean_mm.append(float(d_f2r.mean()))
        p95_mm.append(float(np.percentile(d_f2r, 95)))
        max_mm.append(float(d_f2r.max()))
        cham_mm.append(float(0.5 * (d_f2r.mean() + d_r2f.mean())))
        per_vertex.append(d_f2r)
        if not bool(ok[k]):
            print("  [note] frame %d: forward 시뮬 실패(FAILED) — 오차 신뢰 불가" % fr)

    if not rows:
        return None
    return dict(
        frame_ids=np.array(rows), mean_mm=np.array(mean_mm),
        p95_mm=np.array(p95_mm), max_mm=np.array(max_mm),
        chamfer_mm=np.array(cham_mm), per_vertex=per_vertex,
    )


def write_error_outputs(err):
    fr = err["frame_ids"]; t = (fr - 1) / FPS
    mean_mm, p95_mm, max_mm, cham = (
        err["mean_mm"], err["p95_mm"], err["max_mm"], err["chamfer_mm"])

    csv_path = out(9, "roundtrip_error_per_frame.csv")
    with open(csv_path, "w") as f:
        f.write("frame,time,mean_mm,p95_mm,max_mm,chamfer_mm\n")
        for i in range(len(fr)):
            f.write("%d,%.4f,%.4f,%.4f,%.4f,%.4f\n"
                    % (fr[i], t[i], mean_mm[i], p95_mm[i], max_mm[i], cham[i]))
    print("[out] %s" % csv_path)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, mean_mm, "-o", ms=3, label="mean (fwd→ret)")
    ax.plot(t, p95_mm, "-", lw=1, label="p95")
    ax.plot(t, max_mm, "--", lw=1, alpha=0.7, label="max")
    ax.plot(t, cham, "-", lw=1.2, color="purple", label="chamfer (sym)")
    ax.axhline(mean_mm.mean(), color="gray", ls=":", lw=1,
               label="mean=%.2fmm" % mean_mm.mean())
    ax.set_xlabel("time (s)"); ax.set_ylabel("surface error (mm)")
    ax.set_title("Forward(predicted activations) vs Retarget — surface distance")
    ax.legend(); ax.grid(alpha=0.3)
    png = out(9, "roundtrip_error.png")
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("[out] %s" % png)
    print("[stat] forward vs retarget: mean=%.2fmm  median-of-mean=%.2fmm  "
          "chamfer=%.2fmm  worst-frame max=%.2fmm"
          % (mean_mm.mean(), np.median(mean_mm), cham.mean(), max_mm.max()))


# ----------------------------- (3) 시각 GIF -----------------------------
def make_gif(err=None):
    fwd_v_path = out(8, "forward_surface_verts.npy")
    if not os.path.isfile(fwd_v_path):
        print("[warn] 8_forward_surface_verts 없음 → GIF 건너뜀")
        return
    fwd_v = _forward_to_retarget_frame(np.load(fwd_v_path))      # (T,Nsv,3) mm, aligned
    fwd_f = np.load(out(8, "forward_surface_faces.npy"))
    fframes = np.load(out(8, "forward_frame_ids.npy"))

    retv = retf = None
    rpath = out(4, "retargeted_tongue.npy")
    if os.path.isfile(rpath) and os.path.isfile(TONGUE_OBJ):
        retv = np.load(rpath)                                    # (Tr,Nr,3) mm
        retf = load_faces(TONGUE_OBJ)
    else:
        print("[warn] 4_retargeted_tongue.npy/obj 없음 → retarget 패널 비움")

    masks = None
    mk_col = mk_row = None
    fs = sorted(glob.glob(os.path.join(MRI_ROOT, "mask_*.mat")), key=natkey)
    if fs:
        masks = [sio.loadmat(f)["mask_frame"] for f in fs]
        H = masks[0].shape[0]
        tgt_path = out(1, "tongue_targets.npy")
        if os.path.isfile(tgt_path):
            tgt = np.load(tgt_path)
            mk_col = tgt[..., 0]; mk_row = (H - 1) - tgt[..., 1]
    else:
        print("[warn] MRI 마스크 없음 → MRI 패널 비움")

    # forward + retarget 를 같은 박스(공유 한계)에 그려 직접 비교 가능하게 함
    pts = [fwd_v.reshape(-1, 3)]
    if retv is not None:
        pts.append(retv.reshape(-1, 3))
    P = np.vstack(pts)
    xl = (P[:, 0].min(), P[:, 0].max())
    yl = (P[:, 1].min(), P[:, 1].max())
    zl = (P[:, 2].min(), P[:, 2].max())

    # 오차 색상 스케일(프레임 간 고정)
    err_by_fr = {}
    vmax = None
    if err is not None:
        for i, fr in enumerate(err["frame_ids"]):
            err_by_fr[int(fr)] = err["per_vertex"][i]
        vmax = float(np.percentile(np.concatenate(err["per_vertex"]), 95))

    def set_box(ax, title):
        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
        ax.set_xlabel("x"); ax.set_ylabel("y lat"); ax.set_zlabel("z up")
        ax.view_init(elev=20, azim=-70); ax.set_title(title)

    r0, r1, c0, c1 = 110, 220, 30, 170
    frames_out = []
    T = len(fframes)
    for k in range(0, T, STEP):
        fr = int(fframes[k])
        idx = fr - 1
        fig = plt.figure(figsize=(16, 5))

        # panel 1: MRI
        ax1 = fig.add_subplot(1, 3, 1)
        if masks is not None and 0 <= idx < len(masks):
            ax1.imshow(masks[idx][r0:r1, c0:c1], cmap=CMAP, vmin=0, vmax=6,
                       interpolation="nearest")
            if mk_col is not None and idx < mk_col.shape[0]:
                ax1.plot(mk_col[idx] - c0, mk_row[idx] - r0, "-", c="white", lw=1.2)
                ax1.scatter(mk_col[idx] - c0, mk_row[idx] - r0, c="cyan", s=8, zorder=3)
        ax1.set_title("MRI (tongue=orange) + markers")
        ax1.set_xticks([]); ax1.set_yticks([])

        # panel 2: retarget vs forward OVERLAY (정렬된 같은 박스)
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        if retv is not None and 0 <= idx < len(retv):
            ax2.plot_trisurf(retv[idx][:, 0], retv[idx][:, 1], retv[idx][:, 2],
                             triangles=retf, color="steelblue", alpha=0.45,
                             linewidth=0, edgecolor="none")
        ax2.plot_trisurf(fwd_v[k][:, 0], fwd_v[k][:, 1], fwd_v[k][:, 2],
                         triangles=fwd_f, color="crimson", alpha=0.40,
                         linewidth=0, edgecolor="none")
        terr = ("  mean=%.2fmm" % err["mean_mm"][list(err["frame_ids"]).index(fr)]
                ) if (err is not None and fr in err_by_fr) else ""
        set_box(ax2, "Retarget(blue) vs Forward(red)" + terr)

        # panel 3: forward 표면 위 per-vertex 오차(retarget 까지 거리)
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        if fr in err_by_fr:
            d = err_by_fr[fr]
            sc = ax3.scatter(fwd_v[k][:, 0], fwd_v[k][:, 1], fwd_v[k][:, 2],
                             c=d, cmap="inferno", vmin=0, vmax=vmax, s=8)
            set_box(ax3, "Forward error to retarget (mm)")
            cb = fig.colorbar(sc, ax=ax3, fraction=0.03, pad=0.02)
            cb.ax.tick_params(labelsize=7)
        else:
            ax3.plot_trisurf(fwd_v[k][:, 0], fwd_v[k][:, 1], fwd_v[k][:, 2],
                             triangles=fwd_f, cmap="viridis", alpha=0.9,
                             linewidth=0.1, edgecolor="0.3")
            set_box(ax3, "Forward (from activations)")

        fig.suptitle("MRI vs Retarget vs Forward   frame %d/%d   t=%.1fs @ %gfps"
                     % (fr, int(fframes[-1]), (fr - 1) / FPS, FPS), fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames_out.append(buf); plt.close(fig)

    out_gif = out(9, "compare_forward_vs_retarget_mri.gif")
    imageio.mimsave(out_gif, frames_out, duration=STEP / FPS)
    mb = os.path.getsize(out_gif) / 1e6
    print("[out] %s  (%d frames, %.1f MB, @ %gfps)"
          % (out_gif, len(frames_out), mb, FPS))


def main():
    print_paths()
    err = compute_surface_error()
    if err is not None:
        write_error_outputs(err)
    make_gif(err)


if __name__ == "__main__":
    main()
