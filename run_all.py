#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py - 1~9 파이프라인을 한 번에 순서대로 실행하는 오케스트레이터.

  MRI 마스크 -> (1) 컨투어 -> (2) ArtiSynth 입력 -> (3) lift -> (4) retarget
  -> (5) MRI vs retarget GIF -> (6) 정적 inverse(활성도) -> (7) 활성도 정리
  -> (8) 활성도로 forward -> (9) forward vs retarget 비교

스텝 1~5,7,9 는 순수 파이썬, 스텝 6/8 은 ArtiSynth+JPype(JDK 필요).
6/8 은 --parallel 로 parallel_runner.py(멀티 워커)로 돌릴 수 있고, 이때 PC의
코어/RAM 을 감지해 워커/스레드를 '최대'로 자동 산정한다.

사용 예:
  python3 run_all.py                          # 1~9 전부(현재 MRI_SUBJECT)
  MRI_SUBJECT=Subject2 python3 run_all.py     # 특정 피험자
  python3 run_all.py --parallel               # 6/8 을 자원 최대치로 병렬
  python3 run_all.py --parallel --only 6,8,9  # 무거운 부분만 병렬로
  python3 run_all.py --parallel --xmx 1500m --reserve-gb 2   # 메모리 빡빡할 때
  python3 run_all.py --from 4                 # 4번부터 끝까지
  python3 run_all.py --to 5                   # 1~5까지(순수 파이썬만)
  python3 run_all.py --skip 3,5,7             # 선택 스텝 건너뛰기
  python3 run_all.py --keep-going             # 에러 나도 계속
  python3 run_all.py --dry-run                # 실행할 명령만 출력

튜닝 파라미터(FIT_MODE, INDEPENDENT_FRAMES, N_TARGET_NODES, SETTLE_T, NRAMP,
MAXSTEP, INCOMP, MAX_FRAMES, TONGUE_OBJ, ARTISYNTH_HOME 등)는 환경변수로 주면
각 스텝/워커에 그대로 전달된다.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


# ----------------------- 자원 자동 사이징(6/8 병렬) -----------------------
def _total_ram_gb():
    """전체 RAM(GB). 감지 실패 시 None. (psutil / Linux / Windows)"""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    try:  # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    try:  # Windows
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1e9
    except Exception:
        return None


def _xmx_to_gb(xmx):
    """'2g' / '2048m' / '2000000k' -> GB(float)."""
    s = str(xmx).strip().lower()
    try:
        if s.endswith("g"):
            return float(s[:-1])
        if s.endswith("m"):
            return float(s[:-1]) / 1024
        if s.endswith("k"):
            return float(s[:-1]) / 1024 / 1024
        return float(s) / 1e9
    except Exception:
        return 2.0


def _frame_count():
    """타깃 CSV의 프레임 수(워커 상한). 못 읽으면 None."""
    try:
        import mri_paths
        path = os.path.join(mri_paths.MRI_FIT_DIR, "frame_targets_m.csv")
        seen = set()
        with open(path) as f:
            next(f, None)
            for ln in f:
                if ln.strip():
                    seen.add(ln.split(",")[0])
        return len(seen) or None
    except Exception:
        return None


def auto_resources(xmx, reserve_gb):
    """PC 코어/RAM 기준으로 (workers, threads, note)를 최대로 산정.

    원칙: 워커 수 많게 + 워커당 스레드 적게 (parallel_runner 권장).
      - CPU: 워커 x 스레드 가 논리코어 전부를 쓰도록.
      - RAM: 워커마다 JVM 힙(xmx)을 먹으므로 (가용RAM - reserve)/xmx 로 워커 상한.
      - 프레임 수보다 워커가 많을 필요 없음.
    """
    L = os.cpu_count() or 4
    xmx_gb = _xmx_to_gb(xmx)
    ram = _total_ram_gb()
    if ram:
        usable = max(xmx_gb, ram - reserve_gb)
        w_ram = max(1, int(usable // xmx_gb))
        ram_note = ("RAM %.0fGB - reserve %gGB - %.1fGB/worker => max %d workers"
                    % (ram, reserve_gb, xmx_gb, w_ram))
    else:
        w_ram = L
        ram_note = "RAM unknown => bound by cores only"

    workers = max(1, min(L, w_ram))
    nf = _frame_count()
    if nf:
        workers = min(workers, nf)
    threads = max(1, L // workers)   # 남는 코어는 워커당 스레드로 흡수(전 코어 사용)
    note = ("%s; cores %d => %d workers x %d threads (~%d cores used)"
            % (ram_note, L, workers, threads, workers * threads))
    return workers, threads, note


def _act_csv():
    """6번 산출 활성도 CSV 경로(7번 입력). mri_paths 로 해석(번호 접두사 포함)."""
    try:
        import mri_paths
        return mri_paths.out(6, "activations_static_per_frame.csv")
    except Exception:
        return os.path.join(HERE, "output", "6_activations_static_per_frame.csv")


def build_steps(args):
    """실행할 (번호, 라벨, argv, 필수여부) 목록을 만든다."""
    par = ["--workers", str(args.workers), "--threads", str(args.threads),
           "--xmx", args.xmx]

    if args.parallel:
        step6 = [PY, "parallel_runner.py", "--step", "inverse"] + par
        step8 = [PY, "parallel_runner.py", "--step", "forward"] + par
    else:
        step6 = [PY, "6_static_inverse.py"]
        step8 = [PY, "8_forward_from_activations.py"]

    step7 = [PY, "7_summarize_activations.py", _act_csv(), "--fps", str(args.fps)]
    if args.segments > 0:
        step7 += ["--segments", str(args.segments)]

    # (번호, 라벨, argv, 필수=True / 선택=False)
    return [
        (1, "extract_contours",      [PY, "1_extract_contours.py"],        True),
        (2, "export_artisynth_in",   [PY, "2_export_artisynth_inputs.py"], True),
        (3, "kinematic_lift",        [PY, "3_kinematic_lift.py"],          False),
        (4, "retarget",              [PY, "4_retarget_to_artisynth.py"],   True),
        (5, "compare_mri_retarget",  [PY, "5_compare_gif.py"],             False),
        (6, "static_inverse",        step6,                                True),
        (7, "summarize_activations", step7,                                False),
        (8, "forward_from_acts",     step8,                                True),
        (9, "compare_forward",       [PY, "9_compare_forward.py"],         True),
    ]


def select(steps, args):
    """--from/--to/--only/--skip 적용."""
    nums = set(n for n, *_ in steps)
    if args.only:
        keep = set(int(x) for x in args.only.replace(" ", "").split(",") if x)
    else:
        lo = args.from_ or min(nums)
        hi = args.to or max(nums)
        keep = set(n for n in nums if lo <= n <= hi)
    if args.skip:
        keep -= set(int(x) for x in args.skip.replace(" ", "").split(",") if x)
    return [s for s in steps if s[0] in keep]


def artisynth_ready():
    """6/8 전 사전점검: JPype import + ARTISYNTH_HOME/classes 존재 여부."""
    msgs = []
    try:
        import jpype  # noqa: F401
    except Exception:
        msgs.append("JPype1 not installed (pip install JPype1)")
    ah = os.environ.get("ARTISYNTH_HOME", "/opt/artisynth/artisynth_core")
    if not os.path.isdir(os.path.join(ah, "classes")):
        msgs.append("ARTISYNTH_HOME/classes missing: %s" % ah)
    return msgs


def main():
    ap = argparse.ArgumentParser(
        description="Run the RT-MRI tongue -> ArtiSynth pipeline (steps 1-9).")
    ap.add_argument("--from", dest="from_", type=int, default=None)
    ap.add_argument("--to", type=int, default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--parallel", action="store_true",
                    help="run 6/8 via parallel_runner.py (auto max resources)")
    ap.add_argument("-w", "--workers", type=int, default=None,
                    help="worker count (auto if omitted with --parallel)")
    ap.add_argument("-t", "--threads", type=int, default=None,
                    help="threads per worker (auto if omitted)")
    ap.add_argument("--xmx", default="2g", help="JVM heap per worker (e.g. 2g, 1500m)")
    ap.add_argument("--reserve-gb", dest="reserve_gb", type=float, default=4.0,
                    help="RAM (GB) to leave free for OS; used to size workers")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--segments", type=int, default=7)
    ap.add_argument("--keep-going", dest="keep_going", action="store_true",
                    help="continue even if a required step fails")
    ap.add_argument("--no-clean", dest="no_clean", action="store_true",
                    help="don't wipe output/Subject even on a full run (step 1 included)")
    ap.add_argument("--clean", dest="force_clean", action="store_true",
                    help="force-wipe output/Subject before running (any step range)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    # 병렬이면 워커/스레드 자동 산정(사용자가 -w/-t 명시하면 그 값 우선)
    res_note = None
    if args.parallel and (args.workers is None or args.threads is None):
        aw, at, res_note = auto_resources(args.xmx, args.reserve_gb)
        if args.workers is None:
            args.workers = aw
        if args.threads is None:
            if aw == args.workers:
                args.threads = at
            else:
                args.threads = max(1, (os.cpu_count() or 4) // max(1, args.workers))
    if args.workers is None:
        args.workers = 6
    if args.threads is None:
        args.threads = 2

    steps = select(build_steps(args), args)
    if not steps:
        print("no steps selected (check --only/--skip).")
        return 1

    # 이전 output 정리: 전체 실행(스텝 1 포함) 또는 --clean 일 때만 output/Subject 비움.
    # 부분 실행(--from N 등, 스텝 1 미포함)은 이전 스텝 산출물이 입력이므로 보존.
    is_full_run = any(n == 1 for n, *_ in steps)
    do_clean = (args.force_clean or (is_full_run and not args.no_clean)) and not args.dry_run
    if do_clean:
        try:
            import mri_paths
            print("[clean] wiping previous outputs in %s ..." % mri_paths.MRI_OUT)
            mri_paths.clean_all()
        except Exception as e:
            print("[clean] skipped (%s)" % e)

    subj = os.environ.get("MRI_SUBJECT", "Subject1")
    print("=" * 66)
    print(" pipeline | MRI_SUBJECT=%s | parallel=%s" % (subj, args.parallel))
    if args.parallel:
        print(" resources: %d workers x %d threads x heap %s each"
              % (args.workers, args.threads, args.xmx))
        if res_note:
            print(" sizing: %s" % res_note)
    print(" steps: %s" % ", ".join(str(n) for n, *_ in steps))
    print("=" * 66)

    if any(n in (6, 8) for n, *_ in steps) and not args.dry_run:
        warn = artisynth_ready()
        if warn:
            print("[warn] ArtiSynth check: " + " / ".join(warn))
            print("       (steps 6/8 may fail; use --to 5 for python-only steps)")

    results = []
    t_all = time.time()
    for num, label, argv, required in steps:
        tag = "req" if required else "opt"
        print("\n" + "-" * 66)
        print(">> [%d] %s (%s)" % (num, label, tag))
        print("   $ " + " ".join(argv))
        print("-" * 66, flush=True)
        if args.dry_run:
            results.append((num, label, "DRY", 0.0))
            continue

        t0 = time.time()
        rc = subprocess.call(argv, cwd=HERE, env=os.environ)
        dt = time.time() - t0
        ok = (rc == 0)
        results.append((num, label, "OK" if ok else "FAIL(rc=%d)" % rc, dt))
        if not ok:
            if required and not args.keep_going:
                print("\n[stop] required step %d(%s) failed (rc=%d). "
                      "use --keep-going to continue." % (num, label, rc))
                break
            print("[continue] step %d(%s) failed but proceeding." % (num, label))

    print("\n" + "=" * 66)
    print(" summary (total %.1fs)" % (time.time() - t_all))
    print("=" * 66)
    for num, label, status, dt in results:
        print("  [%d] %-22s %-12s %6.1fs" % (num, label, status, dt))
    failed = [r for r in results if r[2].startswith("FAIL")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
