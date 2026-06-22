#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프레임별 독립 정적 Muscle Inverse (JPype — GUI 없음).

사전 조건:
  - JDK + 컴파일된 ArtiSynth (ARTISYNTH_HOME)
  - pip install JPype1
  - 2_export_artisynth_inputs.py 가 만든 mri_fit/ (manifest + frame_targets_m.csv)

동작 요약 (프레임마다 반복):
  reset → probe 끄기 → 타깃을 NRAMP 단계로 천천히 이동(램프) → hold → 활성도 기록

실행:
  python3 6_static_inverse.py
  INDEPENDENT_FRAMES=0 python3 6_static_inverse.py
"""
import csv
import glob
import os
import sys
import time

ARTISYNTH_HOME = os.environ.get("ARTISYNTH_HOME", "/opt/artisynth/artisynth_core")
MRI_MODEL = os.environ.get("MRI_MODEL", "artisynth.models.tongue3d.FemTongueMriDemo")
MRI_MANIFEST = os.environ.get(
    "MRI_MANIFEST",
    os.path.join(os.path.dirname(__file__), "mri_fit", "mri_fit_tongue.properties"),
)
TARGETS_CSV = os.environ.get(
    "TARGETS_CSV",
    os.path.join(os.path.dirname(__file__), "mri_fit", "frame_targets_m.csv"),
)
OUT_CSV = os.environ.get(
    "OUT_CSV",
    os.path.join(os.path.dirname(__file__), "mri_fit", "activations_static_per_frame.csv"),
)
CONTROLLER = os.environ.get("CONTROLLER", "mriTracking")
SETTLE_T = float(os.environ.get("SETTLE_T", "0.6"))
FPS = float(os.environ.get("FPS", "5.0"))
NRAMP = int(os.environ.get("NRAMP", "30"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))
JVM_XMX = os.environ.get("JVM_XMX", "4g")
MAXSTEP = float(os.environ.get("MAXSTEP", "0.001"))
INCOMP = os.environ.get("INCOMP", "OFF").upper()
MAX_EXCITATION_JUMP = float(os.environ.get("MAX_EXCITATION_JUMP", "0.02"))
INDEPENDENT_FRAMES = os.environ.get("INDEPENDENT_FRAMES", "1").lower() not in ("0", "false", "no")
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "10"))

_S = {
    "main": None,
    "root": None,
    "ctrl": None,
    "tpts": None,
    "exciters": None,
    "names": None,
}


def _log(msg):
    print(msg, flush=True)


def _start_jvm():
    import jpype

    if jpype.isJVMStarted():
        return
    _log("[init] JVM starting (heap %s)..." % JVM_XMX)
    t0 = time.time()
    cp = [os.path.join(ARTISYNTH_HOME, "classes")] + glob.glob(
        os.path.join(ARTISYNTH_HOME, "lib", "*.jar")
    )
    libdir = os.path.join(ARTISYNTH_HOME, "lib")
    jpype.startJVM(
        "-Xmx%s" % JVM_XMX,
        "-Djava.awt.headless=true",
        "-Dartisynth.home=%s" % ARTISYNTH_HOME,
        "-Djava.library.path=%s" % libdir,
        classpath=cp,
    )
    _log("[init] JVM ready (%.1fs)" % (time.time() - t0))


def _jclass(name):
    import jpype
    return jpype.JClass(name)


def _find_tongue(root):
    def rec(m):
        try:
            if hasattr(m, "getMuscleExciters") and m.getMuscleExciters().size() > 0:
                return m
        except Exception:
            pass
        try:
            subs = m.models()
            for i in range(subs.size()):
                r = rec(subs.get(i))
                if r is not None:
                    return r
        except Exception:
            pass
        return None

    tops = root.models()
    for i in range(tops.size()):
        r = rec(tops.get(i))
        if r is not None:
            return r
    return None


def _find_controller(root, name=CONTROLLER):
    try:
        ctrls = root.getControllers()
        try:
            c = ctrls.get(name)
            if c is not None:
                return c
        except Exception:
            pass
        for i in range(ctrls.size()):
            c = ctrls.get(i)
            if c.getClass().getSimpleName() == "TrackingController":
                return c
    except Exception:
        pass
    return None


def _deactivate_probes(root):
    try:
        ips = root.getInputProbes()
        for i in range(ips.size()):
            try:
                ips.get(i).setActive(False)
            except Exception:
                pass
    except Exception:
        pass


def _mean_target_error_mm(tpts, n):
    total, count = 0.0, 0
    for i in range(n):
        tp = tpts.get(i)
        src = tp.getSourceComp()
        if src is None:
            continue
        sp = src.getPosition()
        tp_pos = tp.getPosition()
        dx = sp.x - tp_pos.x
        dz = sp.z - tp_pos.z
        total += (dx * dx + dz * dz) ** 0.5
        count += 1
    return (total / count * 1000.0) if count else float("nan")


def _fit_error_mm(root, tpts, n):
    try:
        return float(root.getTongueFitError()) * 1000.0
    except Exception:
        return _mean_target_error_mm(tpts, n)


def init(manifest=None, model=None):
    """JVM + MRI 모델(manifest) + TrackingController."""
    import jpype

    _start_jvm()
    JString = _jclass("java.lang.String")
    JArray = jpype.JArray
    Main = _jclass("artisynth.core.driver.Main")
    ArrayList = _jclass("java.util.ArrayList")

    manifest = os.path.abspath(manifest or MRI_MANIFEST)
    model = model or MRI_MODEL
    if not os.path.isfile(manifest):
        raise FileNotFoundError("manifest not found: " + manifest)

    _log("[init] loading %s" % model)
    _log("[init] manifest: %s" % manifest)
    t0 = time.time()

    m = Main.getMain()
    if m is None:
        try:
            Main.main(JArray(JString)(["-noGui"]))
        except Exception as e:
            _log("[init] note: Main.main(-noGui): %s" % e)
        m = Main.getMain()
    if m is None:
        m = Main("static_inverse", False)
        m.start(ArrayList())

    args = JArray(JString)([manifest])
    if not m.loadModel(model, model.split(".")[-1], args):
        raise RuntimeError("loadModel failed: " + str(m.getErrorMessage()))

    root = m.getRootModel()
    tongue = _find_tongue(root)
    if tongue is None:
        raise RuntimeError("no FemMuscleModel with exciters found")

    ctrl = _find_controller(root)
    if ctrl is None:
        raise RuntimeError(
            "TrackingController '%s' not found — manifest 확인: %s" % (CONTROLLER, manifest)
        )

    try:
        root.setMaxStepSize(MAXSTEP)
        root.setAdaptiveStepping(False)
        tongue.setGravity(0, 0, 0)
        FemModel = _jclass("artisynth.core.femmodels.FemModel")
        tongue.setIncompressible(getattr(FemModel.IncompMethod, INCOMP))
        tongue.setMaxStepSize(MAXSTEP)
        _log("[init] incompressible=%s, maxStep=%.4f" % (INCOMP, MAXSTEP))
    except Exception as e:
        _log("[init] note: FEM stability tweak failed: %s" % e)

    try:
        ctrl.setMaxExcitationJump(MAX_EXCITATION_JUMP)
    except Exception:
        pass

    exlist = tongue.getMuscleExciters()
    exciters = [exlist.get(i) for i in range(exlist.size())]
    names = [str(e.getName()) for e in exciters]
    tpts = ctrl.getTargetPoints()

    _S.update(
        main=m, root=root, ctrl=ctrl, tpts=tpts, exciters=exciters, names=names,
    )
    mode = "independent (reset/frame)" if INDEPENDENT_FRAMES else "continuous (no reset)"
    _log(
        "[init] ready in %.1fs — %d targets, %d exciters"
        % (time.time() - t0, tpts.size(), len(exciters))
    )
    _log("[init] mode: %s, settle=%.2fs, nramp=%d" % (mode, SETTLE_T, NRAMP))
    return list(names)


def load_targets(path=None):
    """frame_targets_m.csv → {프레임번호: {노드인덱스: (x,y,z)}}."""
    path = path or TARGETS_CSV
    frames = {}
    with open(path, newline="") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if not row:
                continue
            fr = int(row[0])
            idx = int(row[1])
            frames.setdefault(fr, {})[idx] = (
                float(row[2]), float(row[3]), float(row[4]),
            )
    return frames


def _write_csv(path, names, records, fps):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time"] + names)
        for fr, acts in records:
            t = (fr - 1) / fps
            wr.writerow([fr, "%.4f" % t] + ["%.5f" % acts.get(nm, 0.0) for nm in names])


def _rest_targets(tpts, n):
    rest = {}
    for i in range(n):
        tp = tpts.get(i)
        src = tp.getSourceComp()
        if src is not None:
            p = src.getPosition()
            rest[i] = (p.x, p.y, p.z)
    return rest


def _set_targets_blend(tpts, rest, goals, frac, n):
    Point3d = _jclass("maspack.matrix.Point3d")
    for i in range(n):
        if i not in goals:
            continue
        r = rest.get(i, goals[i])
        g = goals[i]
        tpts.get(i).setPosition(Point3d(
            r[0] + frac * (g[0] - r[0]),
            r[1] + frac * (g[1] - r[1]),
            r[2] + frac * (g[2] - r[2]),
        ))


def _settle_ramped(m, tpts, goals, n, label=""):
    nramp = max(1, NRAMP)
    ramp_t = SETTLE_T * 0.75
    hold_t = SETTLE_T * 0.5
    seg = ramp_t / nramp
    rest = _rest_targets(tpts, n)

    _log("  %sramp targets %d steps (%.2fs) + hold %.2fs..." % (
        ("%s — " % label) if label else "", nramp, ramp_t, hold_t,
    ))
    t0 = time.time()
    ok = True

    for k in range(1, nramp + 1):
        frac = float(k) / nramp
        _set_targets_blend(tpts, rest, goals, frac, n)
        m.playAndWait(seg * k)
        ex = m.getSimulationException()
        if ex is not None:
            _log("  FAILED at ramp %d/%d: %s" % (k, nramp, ex))
            ok = False
            break

    if ok:
        _set_targets_blend(tpts, rest, goals, 1.0, n)
        m.playAndWait(ramp_t + hold_t)
        ex = m.getSimulationException()
        if ex is not None:
            _log("  FAILED at hold: %s" % ex)
            ok = False

    return ok, time.time() - t0


def solve_frame(targets, settle=None, frame_no=None, reset=None):
    """한 프레임 inverse → (activations dict, fit_error_mm, ok bool)."""
    if _S["main"] is None:
        init()
    if settle is None:
        settle = SETTLE_T
    if reset is None:
        reset = INDEPENDENT_FRAMES

    m, root, tpts, exciters = _S["main"], _S["root"], _S["tpts"], _S["exciters"]
    n = min(tpts.size(), 11)
    label = ("frame %d" % frame_no) if frame_no is not None else "frame"

    if reset:
        m.reset()
    _deactivate_probes(root)

    ok, elapsed = _settle_ramped(m, tpts, targets, n, label=label)
    acts = {str(e.getName()): float(e.getExcitation()) for e in exciters}
    fit_mm = _fit_error_mm(root, tpts, n)
    max_act = max(acts.values()) if acts else 0.0
    status = "OK" if ok else "FAILED (inverted elements — 결과 무의미)"
    _log(
        "  %s — %s in %.1fs | fit_err=%.2f mm | max_act=%.3f"
        % (label, status, elapsed, fit_mm, max_act)
    )
    if not ok:
        _log("  hint: SETTLE_T↑ NRAMP↑ MAX_EXCITATION_JUMP↓ 또는 INDEPENDENT_FRAMES=1")
    return acts, fit_mm, ok


def run_static_inverse(
    targets_csv=None,
    out_csv=None,
    settle=None,
    fps=None,
    max_frames=None,
):
    if _S["main"] is None:
        init()

    targets_csv = targets_csv or TARGETS_CSV
    out_csv = out_csv or OUT_CSV
    settle = SETTLE_T if settle is None else settle
    fps = FPS if fps is None else fps
    max_frames = MAX_FRAMES if max_frames is None else max_frames

    frames = load_targets(targets_csv)
    frame_ids = sorted(frames.keys())
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]

    names = _S["names"]
    total = len(frame_ids)
    _log("Loaded %d frames from %s" % (total, targets_csv))
    _log("Output -> %s" % out_csv)

    records = []
    n = min(_S["tpts"].size(), 11)
    t_run = time.time()

    for k, fr in enumerate(frame_ids, 1):
        pct = 100.0 * k / total
        t_mri = (fr - 1) / fps
        eta = ""
        if k > 1:
            avg = (time.time() - t_run) / (k - 1)
            eta = " | ETA ~%.0fs" % (avg * (total - k + 1))
        _log("[%d/%d] %.0f%% frame=%d t=%.2fs%s" % (k, total, pct, fr, t_mri, eta))

        do_reset = INDEPENDENT_FRAMES or (k == 1)
        goals = {i: frames[fr][i] for i in range(n) if i in frames[fr]}
        acts, fit_mm, ok = solve_frame(
            goals, settle=settle, frame_no=fr, reset=do_reset,
        )
        if not ok and not INDEPENDENT_FRAMES:
            _log("  -> mesh corrupted; use INDEPENDENT_FRAMES=1 or fix params")
        records.append((fr, acts))

        top = sorted(acts.items(), key=lambda kv: -kv[1])[:3]
        _log("  top: " + ", ".join("%s=%.3f" % (nm, v) for nm, v in top))

        if CHECKPOINT_EVERY > 0 and k % CHECKPOINT_EVERY == 0:
            ckpt = out_csv.replace(".csv", "_checkpoint.csv")
            _write_csv(ckpt, names, records, fps)
            _log("[checkpoint] %d frames -> %s" % (k, ckpt))

    _write_csv(out_csv, names, records, fps)
    _log(
        "DONE. %d frames in %.1fs -> %s"
        % (len(records), time.time() - t_run, out_csv)
    )
    return out_csv


def muscle_names():
    if _S["names"] is None:
        init()
    return list(_S["names"])


def shutdown():
    import jpype
    if jpype.isJVMStarted():
        jpype.shutdownJVM()
    _S.update(main=None, root=None, ctrl=None, tpts=None, exciters=None, names=None)


if __name__ == "__main__":
    try:
        init()
        run_static_inverse()
    except KeyboardInterrupt:
        _log("interrupted")
        sys.exit(130)
    finally:
        shutdown()
