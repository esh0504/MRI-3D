#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8_forward_from_activations.py

Round-trip 1단계: inverse(6번)가 구한 '근육 활성도'로 ArtiSynth FEM 혀를 실제로
forward 구동해서 3D Tongue Motion을 추출한다.

흐름:
  activations_static_per_frame.csv (6번 출력)
     → 프레임마다 artisynth_forward.apply_activation(활성도)
     → FEM 노드 위치(read_nodes) + 표면 메쉬(read_surface) 기록
     → forward_*.npy 저장 (9번 비교 입력)

비교가 깨끗한 이유: forward(HexTongueDemo)와 inverse(FemTongueMriDemo)는 같은 FEM
혀(같은 노드 번호)를 쓴다. 그래서 forward 노드 위치를 inverse가 저장한 goals(_goals.npz)
의 node_numbers로 그대로 매칭해 오차를 잴 수 있다(단위 metres 동일).

환경변수:
  MRI_SUBJECT, MRI_OUT     경로(공통 mri_paths)
  ACT_CSV                  활성도 CSV (기본: MRI_OUT/activations_static_per_frame.csv)
  SETTLE_T, NRAMP, MAXSTEP, INCOMP   forward 시뮬 파라미터(6번과 맞출 것)
  MAX_FRAMES               앞에서부터 N프레임만(0=전체)
  EXPORT_OBJ_EVERY         N프레임마다 .obj 저장(0=안 함)

실행:
  python3 8_forward_from_activations.py
"""
import csv
import os
import sys
import time

import numpy as np

from mri_paths import MRI_OUT, out, print_paths
import artisynth_forward as fwd

ACT_CSV = os.environ.get(
    "ACT_CSV", out(6, "activations_static_per_frame.csv"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))
EXPORT_OBJ_EVERY = int(os.environ.get("EXPORT_OBJ_EVERY", "0"))
# 병렬: 이 워커가 맡을 프레임만 처리(frame_ids[FRAME_WORKER::FRAME_NWORKERS]).
FRAME_NWORKERS = int(os.environ.get("FRAME_NWORKERS", "1"))
FRAME_WORKER = int(os.environ.get("FRAME_WORKER", "0"))


def load_activations(path):
    """6번 CSV → (frame_ids[T], names[M], acts[T,M])."""
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        names = header[2:]            # frame,time,<muscles...>
        frame_ids, rows = [], []
        for row in rd:
            if not row:
                continue
            frame_ids.append(int(row[0]))
            rows.append([float(x) for x in row[2:]])
    return np.array(frame_ids, dtype=int), names, np.array(rows, dtype=float)


def main():
    print_paths()
    if not os.path.isfile(ACT_CSV):
        raise SystemExit(
            "활성도 CSV 없음: %s\n  먼저 6_static_inverse.py 를 실행하세요." % ACT_CSV)

    frame_ids, names, acts = load_activations(ACT_CSV)
    if MAX_FRAMES > 0:
        frame_ids, acts = frame_ids[:MAX_FRAMES], acts[:MAX_FRAMES]
    if FRAME_NWORKERS > 1:
        sel = slice(FRAME_WORKER, None, FRAME_NWORKERS)
        frame_ids, acts = frame_ids[sel], acts[sel]
        print("[worker %d/%d] %d frames" % (FRAME_WORKER, FRAME_NWORKERS, len(frame_ids)))
    T = len(frame_ids)
    print("[fwd] %d frames, %d muscles from %s" % (T, len(names), ACT_CSV))

    fwd_names = fwd.init()
    print("[fwd] muscle order(model): %s" % ",".join(fwd_names))

    obj_dir = out(8, "forward_objs")
    if EXPORT_OBJ_EVERY > 0:
        os.makedirs(obj_dir, exist_ok=True)

    node_pos_all = None
    node_numbers = None
    surf_verts_all = None
    surf_faces = None
    ok_flags = []

    t_run = time.time()
    for k, fr in enumerate(frame_ids):
        act_dict = {nm: float(acts[k, j]) for j, nm in enumerate(names)}
        ok = fwd.apply_activation(act_dict)
        npos, nnum = fwd.read_nodes()
        sverts, sfaces = fwd.read_surface()

        if node_pos_all is None:
            node_numbers = nnum
            surf_faces = sfaces
            node_pos_all = np.empty((T, npos.shape[0], 3))
            surf_verts_all = np.empty((T, sverts.shape[0], 3))
        node_pos_all[k] = npos
        surf_verts_all[k] = sverts
        ok_flags.append(ok)

        maxa = float(np.max(acts[k])) if acts.shape[1] else 0.0
        eta = ""
        if k > 0:
            avg = (time.time() - t_run) / k
            eta = " | ETA ~%.0fs" % (avg * (T - k))
        print("[%d/%d] frame=%d %s max_act=%.3f%s"
              % (k + 1, T, fr, "OK" if ok else "FAILED", maxa, eta), flush=True)

        if EXPORT_OBJ_EVERY > 0 and (k % EXPORT_OBJ_EVERY == 0):
            fwd.save_obj(sverts, sfaces,
                         os.path.join(obj_dir, "forward_%03d.obj" % fr))

    # ---- 저장 ----
    os.makedirs(MRI_OUT, exist_ok=True)
    np.save(out(8, "forward_nodes.npy"), node_pos_all)
    np.save(out(8, "forward_node_numbers.npy"), node_numbers)
    np.save(out(8, "forward_surface_verts.npy"), surf_verts_all)
    np.save(out(8, "forward_surface_faces.npy"), surf_faces)
    np.save(out(8, "forward_frame_ids.npy"), frame_ids)
    np.save(out(8, "forward_ok.npy"), np.array(ok_flags, dtype=bool))

    nfail = int(np.sum(~np.array(ok_flags)))
    print("[out] 8_forward_nodes.npy %s  (metres)" % (node_pos_all.shape,))
    print("[out] 8_forward_surface_verts.npy %s + faces %s"
          % (surf_verts_all.shape, surf_faces.shape))
    print("DONE. %d frames in %.1fs (%d failed) -> %s"
          % (T, time.time() - t_run, nfail, MRI_OUT))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.flush()
        os._exit(130)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
