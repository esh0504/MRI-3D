#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9_compare_forward.py

Round-trip 2·3단계: forward(8번)로 뽑은 3D Tongue Motion을
  (2) Retargeting motion 과 정량 비교  → 노드별 오차 그래프/CSV
  (3) MRI 영상과 함께 3분할 GIF 로 시각 비교

(2) 정량 비교 (option A: FEM 타깃 노드 기준):
  forward 노드 위치 ↔ inverse(6번)가 저장한 goals(_goals.npz, = retarget 목표 위치)
  를 '같은 FEM 노드 번호'로 매칭해 거리 오차(mm)를 잰다. 같은 모델·같은 노드·같은
  단위(metres)라 메쉬 대응/스케일 문제가 없다.
  -> 작으면 "활성도가 retarget 형상을 실제로 재현한다"(왕복이 닫힘)는 뜻.

(3) 시각 비교 GIF (프레임 동기):
  [MRI 마스크 + 추적 마커] | [Retarget 메쉬] | [Forward 메쉬]

환경변수: MRI_SUBJECT, MRI_ROOT, MRI_OUT (공통), STEP(gif 프레임 간격)

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

from mri_paths import MRI_ROOT, MRI_OUT, TONGUE_OBJ, print_paths

ACT_CSV = os.path.join(MRI_OUT, "activations_static_per_frame.csv")
GOALS_NPZ = os.path.join(MRI_OUT, "activations_static_per_frame_goals.npz")
FPS = float(os.environ.get("FPS", "5.0"))
STEP = int(os.environ.get("STEP", "1"))
CMAP = ListedColormap(["black", "red", "green", "blue", "orange", "purple", "skyblue"])
M2MM = 1000.0


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


# ----------------------------- (2) 정량 오차 -----------------------------
def compute_node_error():
    """forward 노드 vs retarget goal 오차. 반환: (frame_ids, mean_mm, max_mm, p95_mm)
    또는 입력이 없으면 None."""
    if not (os.path.isfile(GOALS_NPZ)
            and os.path.isfile(os.path.join(MRI_OUT, "forward_nodes.npy"))):
        print("[warn] goals 또는 forward_nodes 없음 → 정량 오차 건너뜀")
        return None

    g = np.load(GOALS_NPZ)
    gframes, gnodes, goals = g["frame_ids"], g["node_numbers"], g["goals"]  # (Tg,),(n,),(Tg,n,3)
    fpos = np.load(os.path.join(MRI_OUT, "forward_nodes.npy"))               # (T,Nn,3)
    fnum = np.load(os.path.join(MRI_OUT, "forward_node_numbers.npy"))        # (Nn,)
    fframes = np.load(os.path.join(MRI_OUT, "forward_frame_ids.npy"))        # (T,)

    col = {int(num): i for i, num in enumerate(fnum)}          # 노드번호 -> forward 열
    frow = {int(fr): i for i, fr in enumerate(fframes)}        # frame id -> forward 행
    cols = np.array([col.get(int(nn), -1) for nn in gnodes])
    valid = cols >= 0
    if not valid.all():
        print("[warn] %d/%d 타깃 노드가 forward에 없음(매칭 실패)"
              % (int((~valid).sum()), len(gnodes)))

    rows, mean_mm, max_mm, p95_mm = [], [], [], []
    for gi, fr in enumerate(gframes):
        fi = frow.get(int(fr))
        if fi is None:
            continue
        fp = fpos[fi][cols[valid]]            # (n_valid,3) forward
        gp = goals[gi][valid]                 # (n_valid,3) retarget goal
        d = np.linalg.norm(fp - gp, axis=1) * M2MM
        rows.append(int(fr))
        mean_mm.append(float(d.mean()))
        max_mm.append(float(d.max()))
        p95_mm.append(float(np.percentile(d, 95)))

    return (np.array(rows), np.array(mean_mm), np.array(max_mm), np.array(p95_mm))


def write_error_outputs(err):
    frame_ids, mean_mm, max_mm, p95_mm = err
    t = (frame_ids - 1) / FPS
    csv_path = os.path.join(MRI_OUT, "roundtrip_error_per_frame.csv")
    with open(csv_path, "w") as f:
        f.write("frame,time,mean_mm,p95_mm,max_mm\n")
        for i in range(len(frame_ids)):
            f.write("%d,%.4f,%.4f,%.4f,%.4f\n"
                    % (frame_ids[i], t[i], mean_mm[i], p95_mm[i], max_mm[i]))
    print("[out] %s" % csv_path)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, mean_mm, "-o", ms=3, label="mean")
    ax.plot(t, p95_mm, "-", lw=1, label="p95")
    ax.plot(t, max_mm, "--", lw=1, alpha=0.7, label="max")
    ax.axhline(mean_mm.mean(), color="gray", ls=":", lw=1,
               label="mean=%.2fmm" % mean_mm.mean())
    ax.set_xlabel("time (s)"); ax.set_ylabel("node error (mm)")
    ax.set_title("Round-trip error: forward(activations) vs retarget goal")
    ax.legend(); ax.grid(alpha=0.3)
    png = os.path.join(MRI_OUT, "roundtrip_error.png")
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("[out] %s" % png)
    print("[stat] round-trip error: mean=%.2fmm  median-of-mean=%.2fmm  worst-frame max=%.2fmm"
          % (mean_mm.mean(), np.median(mean_mm), max_mm.max()))


# ----------------------------- (3) 시각 GIF -----------------------------
def make_gif():
    fwd_v_path = os.path.join(MRI_OUT, "forward_surface_verts.npy")
    if not os.path.isfile(fwd_v_path):
        print("[warn] forward_surface_verts 없음 → GIF 건너뜀")
        return
    fwd_v = np.load(fwd_v_path) * M2MM                       # (T,Nsv,3) mm
    fwd_f = np.load(os.path.join(MRI_OUT, "forward_surface_faces.npy"))
    fframes = np.load(os.path.join(MRI_OUT, "forward_frame_ids.npy"))

    retv = retf = None
    rpath = os.path.join(MRI_OUT, "retargeted_tongue.npy")
    if os.path.isfile(rpath) and os.path.isfile(TONGUE_OBJ):
        retv = np.load(rpath)                                # (Tr,433,3) mm
        retf = load_faces(TONGUE_OBJ)
    else:
        print("[warn] retargeted_tongue.npy/obj 없음 → retarget 패널 비움")

    masks = None
    mk_col = mk_row = None
    fs = sorted(glob.glob(os.path.join(MRI_ROOT, "mask_*.mat")), key=natkey)
    if fs:
        masks = [sio.loadmat(f)["mask_frame"] for f in fs]
        H = masks[0].shape[0]
        tgt_path = os.path.join(MRI_OUT, "tongue_targets.npy")
        if os.path.isfile(tgt_path):
            tgt = np.load(tgt_path)
            mk_col = tgt[..., 0]; mk_row = (H - 1) - tgt[..., 1]
    else:
        print("[warn] MRI 마스크 없음 → MRI 패널 비움")

    def lims(arr):
        P = arr.reshape(-1, 3)
        return ((P[:, 0].min(), P[:, 0].max()),
                (P[:, 1].min(), P[:, 1].max()),
                (P[:, 2].min(), P[:, 2].max()))
    fxl, fyl, fzl = lims(fwd_v)
    if retv is not None:
        rxl, ryl, rzl = lims(retv)

    def draw_mesh(ax, V, F, xl, yl, zl, title):
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                        cmap="viridis", alpha=0.9, linewidth=0.1, edgecolor="0.3")
        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
        ax.set_xlabel("x"); ax.set_ylabel("y lat"); ax.set_zlabel("z up")
        ax.view_init(elev=20, azim=-70); ax.set_title(title)

    r0, r1, c0, c1 = 110, 220, 30, 170
    frames_out = []
    T = len(fframes)
    for k in range(0, T, STEP):
        fr = int(fframes[k])
        idx = fr - 1                                          # MRI/retarget 0-based
        fig = plt.figure(figsize=(15, 5))

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

        # panel 2: retarget mesh
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        if retv is not None and 0 <= idx < len(retv):
            draw_mesh(ax2, retv[idx], retf, rxl, ryl, rzl, "Retargeted tongue")
        else:
            ax2.set_title("Retargeted (n/a)")

        # panel 3: forward mesh
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        draw_mesh(ax3, fwd_v[k], fwd_f, fxl, fyl, fzl, "Forward (from activations)")

        fig.suptitle("MRI vs Retarget vs Forward   frame %d/%d   t=%.1fs @ %gfps"
                     % (fr, int(fframes[-1]), (fr - 1) / FPS, FPS), fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames_out.append(buf); plt.close(fig)

    out = os.path.join(MRI_OUT, "compare_forward_vs_retarget_mri.gif")
    imageio.mimsave(out, frames_out, duration=STEP / FPS)
    mb = os.path.getsize(out) / 1e6
    print("[out] %s  (%d frames, %.1f MB, @ %gfps)"
          % (out, len(frames_out), mb, FPS))


def main():
    print_paths()
    err = compute_node_error()
    if err is not None:
        write_error_outputs(err)
    make_gif()


if __name__ == "__main__":
    main()
