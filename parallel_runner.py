#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parallel_runner.py

프레임 단위 병렬 실행기. 프레임이 서로 독립(INDEPENDENT_FRAMES)이라 여러 워커
프로세스(각자 JVM)가 프레임을 나눠 동시에 풀고, 끝나면 결과를 하나로 합친다.
GPU 없이 멀티코어 CPU로 inverse(6번)/forward(8번)를 크게 가속한다.

각 워커는 frame_ids[w::W]만 처리(라운드로빈). Pardiso(MKL) 스레드는 워커당 적게
주고(THREADS_PER_WORKER) 워커 수를 늘리는 편이 총 처리량이 좋다.

사용:
  python3 parallel_runner.py                      # inverse, 기본 워커수
  python3 parallel_runner.py --step inverse -w 6 -t 2
  python3 parallel_runner.py --step forward -w 8 -t 2

옵션(환경변수로도 가능):
  --step      inverse | forward            (기본 inverse)
  -w/--workers   워커 프로세스 수           (기본 6)
  -t/--threads   워커당 MKL 스레드 수       (기본 2)
  --xmx          워커당 JVM 힙              (기본 2g)
  --keep-parts   중간 파트 파일 보존(기본 삭제)
튜닝 파라미터(FIT_MODE, N_TARGET_NODES, SETTLE_T, NRAMP, MAXSTEP, INCOMP,
MAX_FRAMES, MRI_SUBJECT 등)는 환경변수로 주면 워커에 그대로 전달된다.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import mri_paths

HERE = os.path.dirname(os.path.abspath(__file__))


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["inverse", "forward"],
                    default=os.environ.get("STEP", "inverse"))
    ap.add_argument("-w", "--workers", type=int,
                    default=_env_int("NWORKERS", 6))
    ap.add_argument("-t", "--threads", type=int,
                    default=_env_int("THREADS_PER_WORKER", 2))
    ap.add_argument("--xmx", default=os.environ.get("JVM_XMX", "2g"))
    ap.add_argument("--keep-parts", action="store_true")
    return ap.parse_args()


def _worker_env(base_threads, xmx):
    """워커 공통 환경: 스레드 제한 + JVM 힙. (os.environ 상속 위에 덮어씀)"""
    env = os.environ.copy()
    for k in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = str(base_threads)
    env["JVM_XMX"] = xmx
    env["CHECKPOINT_EVERY"] = "0"
    # 잘못된 잔재 env가 워커 경로를 오염시키지 않도록 명시 경로만 쓰게 제거
    for k in ("OUT_CSV", "TARGETS_CSV", "MRI_MANIFEST", "ACT_CSV"):
        env.pop(k, None)
    return env


def _launch(cmd, env, logpath):
    log = open(logpath, "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            cwd=HERE), log


def _wait_all(procs, logs, label):
    """모든 워커가 끝날 때까지 대기하며 진행상황 출력. 반환: 실패한 워커 인덱스 목록."""
    t0 = time.time()
    n = len(procs)
    done = [False] * n
    while not all(done):
        time.sleep(5)
        ndone = 0
        for i, p in enumerate(procs):
            if p.poll() is not None:
                done[i] = True
            ndone += int(done[i])
        print("  [%s] %d/%d workers done | %.0fs elapsed"
              % (label, ndone, n, time.time() - t0), flush=True)
    for lg in logs:
        try:
            lg.close()
        except Exception:
            pass
    failed = [i for i, p in enumerate(procs) if p.returncode != 0]
    return failed


def _tail(path, k=25):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-k:])
    except Exception:
        return "(no log)"


# ----------------------------- inverse -----------------------------
def run_inverse(W, threads, xmx, parts, keep):
    targets = os.path.join(mri_paths.MRI_FIT_DIR, "frame_targets_m.csv")
    manifest = os.path.join(mri_paths.MRI_FIT_DIR, "mri_fit_tongue.properties")
    final_out = mri_paths.out(6, "activations_static_per_frame.csv")
    if not os.path.isfile(targets):
        raise SystemExit("타깃 CSV 없음: %s (먼저 2_export 실행)" % targets)

    procs, logs, part_csvs = [], [], []
    print("[inverse] launching %d workers x %d threads (xmx %s)" % (W, threads, xmx))
    for w in range(W):
        env = _worker_env(threads, xmx)
        env["FRAME_NWORKERS"] = str(W)
        env["FRAME_WORKER"] = str(w)
        env["TARGETS_CSV"] = targets
        env["MRI_MANIFEST"] = manifest
        part = os.path.join(parts, "inv_w%d.csv" % w)
        env["OUT_CSV"] = part
        part_csvs.append(part)
        cmd = [sys.executable, "-u", os.path.join(HERE, "6_static_inverse.py")]
        p, lg = _launch(cmd, env, os.path.join(parts, "inv_w%d.log" % w))
        procs.append(p); logs.append(lg)

    failed = _wait_all(procs, logs, "inverse")
    if failed:
        for w in failed:
            print("\n[inverse] WORKER %d FAILED — log tail:\n%s"
                  % (w, _tail(os.path.join(parts, "inv_w%d.log" % w))), file=sys.stderr)
        raise SystemExit("inverse: %d workers failed" % len(failed))

    _merge_csv(part_csvs, final_out)
    _merge_goals([p.replace(".csv", "_goals.npz") for p in part_csvs],
                 final_out.replace(".csv", "_goals.npz"))
    if not keep:
        shutil.rmtree(parts, ignore_errors=True)
    print("[inverse] DONE -> %s" % final_out)


def _merge_csv(part_csvs, final_out):
    header = None
    rows = {}
    for pc in part_csvs:
        if not os.path.isfile(pc):
            continue
        with open(pc) as f:
            lines = f.read().splitlines()
        if not lines:
            continue
        header = header or lines[0]
        for ln in lines[1:]:
            if ln.strip():
                fr = int(ln.split(",")[0])
                rows[fr] = ln
    if header is None:
        raise SystemExit("merge: 워커 출력이 비어있음")
    os.makedirs(os.path.dirname(os.path.abspath(final_out)) or ".", exist_ok=True)
    with open(final_out, "w") as f:
        f.write(header + "\n")
        for fr in sorted(rows):
            f.write(rows[fr] + "\n")
    print("[merge] %d frames -> %s" % (len(rows), final_out))


def _merge_goals(part_npzs, final_npz):
    node_numbers = None
    frame_to_goal = {}
    for pn in part_npzs:
        if not os.path.isfile(pn):
            continue
        d = np.load(pn)
        if node_numbers is None:
            node_numbers = d["node_numbers"]
        for fr, g in zip(d["frame_ids"], d["goals"]):
            frame_to_goal[int(fr)] = g
    if node_numbers is None or not frame_to_goal:
        print("[merge] goals 없음(midsag2d?) → 건너뜀")
        return
    frames = sorted(frame_to_goal)
    goals = np.array([frame_to_goal[fr] for fr in frames])
    np.savez(final_npz, frame_ids=np.array(frames, dtype=int),
             node_numbers=node_numbers, goals=goals)
    print("[merge] goals %d frames -> %s" % (len(frames), final_npz))


# ----------------------------- forward -----------------------------
def run_forward(W, threads, xmx, parts, keep):
    act_csv = os.environ.get(
        "ACT_CSV", mri_paths.out(6, "activations_static_per_frame.csv"))
    if not os.path.isfile(act_csv):
        raise SystemExit("활성도 CSV 없음: %s (먼저 inverse 실행)" % act_csv)

    procs, logs, part_dirs = [], [], []
    print("[forward] launching %d workers x %d threads (xmx %s)" % (W, threads, xmx))
    for w in range(W):
        env = _worker_env(threads, xmx)
        env["FRAME_NWORKERS"] = str(W)
        env["FRAME_WORKER"] = str(w)
        env["ACT_CSV"] = act_csv
        wdir = os.path.join(parts, "fwd_w%d" % w)
        os.makedirs(wdir, exist_ok=True)
        env["MRI_OUT"] = wdir          # 8번이 이 디렉터리에 forward_*.npy 기록
        part_dirs.append(wdir)
        cmd = [sys.executable, "-u", os.path.join(HERE, "8_forward_from_activations.py")]
        p, lg = _launch(cmd, env, os.path.join(parts, "fwd_w%d.log" % w))
        procs.append(p); logs.append(lg)

    failed = _wait_all(procs, logs, "forward")
    if failed:
        for w in failed:
            print("\n[forward] WORKER %d FAILED — log tail:\n%s"
                  % (w, _tail(os.path.join(parts, "fwd_w%d.log" % w))), file=sys.stderr)
        raise SystemExit("forward: %d workers failed" % len(failed))

    _merge_forward(part_dirs, mri_paths.MRI_OUT)
    if not keep:
        shutil.rmtree(parts, ignore_errors=True)
    print("[forward] DONE -> %s" % mri_paths.MRI_OUT)


def _merge_forward(part_dirs, out_dir):
    # 워커는 MRI_OUT=wdir 로 돌아 wdir/8_forward_*.npy 를 만든다(8번 번호 규칙).
    node_numbers = faces = None
    recs = {}   # frame_id -> (nodes(Nn,3), surf(Nsv,3), ok)
    for wdir in part_dirs:
        fp = os.path.join(wdir, "8_forward_frame_ids.npy")
        if not os.path.isfile(fp):
            continue
        fids = np.load(fp)
        nodes = np.load(os.path.join(wdir, "8_forward_nodes.npy"))
        surf = np.load(os.path.join(wdir, "8_forward_surface_verts.npy"))
        ok = np.load(os.path.join(wdir, "8_forward_ok.npy"))
        if node_numbers is None:
            node_numbers = np.load(os.path.join(wdir, "8_forward_node_numbers.npy"))
            faces = np.load(os.path.join(wdir, "8_forward_surface_faces.npy"))
        for i, fr in enumerate(fids):
            recs[int(fr)] = (nodes[i], surf[i], bool(ok[i]))
    if not recs:
        raise SystemExit("merge: forward 워커 출력이 비어있음")
    frames = sorted(recs)
    nodes_all = np.stack([recs[fr][0] for fr in frames], 0)
    surf_all = np.stack([recs[fr][1] for fr in frames], 0)
    ok_all = np.array([recs[fr][2] for fr in frames], dtype=bool)
    os.makedirs(out_dir, exist_ok=True)
    np.save(mri_paths.out(8, "forward_nodes.npy"), nodes_all)
    np.save(mri_paths.out(8, "forward_node_numbers.npy"), node_numbers)
    np.save(mri_paths.out(8, "forward_surface_verts.npy"), surf_all)
    np.save(mri_paths.out(8, "forward_surface_faces.npy"), faces)
    np.save(mri_paths.out(8, "forward_frame_ids.npy"), np.array(frames, dtype=int))
    np.save(mri_paths.out(8, "forward_ok.npy"), ok_all)
    print("[merge] forward %d frames (%d failed) -> %s"
          % (len(frames), int((~ok_all).sum()), out_dir))


def main():
    a = parse_args()
    mri_paths.print_paths()
    W = max(1, a.workers)
    parts = os.path.join(mri_paths.MRI_OUT, ".parallel_parts_%s" % a.step)
    if os.path.isdir(parts):
        shutil.rmtree(parts, ignore_errors=True)
    os.makedirs(parts, exist_ok=True)
    t0 = time.time()
    if a.step == "inverse":
        run_inverse(W, a.threads, a.xmx, parts, a.keep_parts)
    else:
        run_forward(W, a.threads, a.xmx, parts, a.keep_parts)
    print("ALL DONE in %.1fs (step=%s, workers=%d x %d threads)"
          % (time.time() - t0, a.step, W, a.threads))


if __name__ == "__main__":
    main()
