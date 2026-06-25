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

Fitting 모드 (FIT_MODE):
  surface3d (기본): MRI 정중시상 '변위'(프레임 - rest프레임)를 대칭 RBF로 FEM
      표면 노드 다수에 입혀 3D 타깃 생성. 절대좌표가 아닌 상대변위라 rest
      프레임에서 오차 0 → inverted element blow-up 회피. subWeights=(1,1,1).
      => "3D 형상을 어떤 근육값으로 만들 수 있는가"의 inverse.
  midsag2d: 기존 방식. dorsal 11개 노드를 MRI 윤곽 절대좌표에 맞춤(과소결정,
      정합 오차로 blow-up 위험).

실행:
  python3 6_static_inverse.py
  FIT_MODE=midsag2d python3 6_static_inverse.py
  N_TARGET_NODES=80 SUBW=1,0.3,1 python3 6_static_inverse.py
  INDEPENDENT_FRAMES=0 python3 6_static_inverse.py
"""
import csv
import glob
import os
import sys
import time

from mri_paths import MRI_FIT_DIR, MRI_OUT, print_paths

# =========================================================================
# 하이퍼파라미터 (모두 환경변수로 덮어쓸 수 있음: VAR=값 python3 6_static_inverse.py)
# =========================================================================

# --- 경로 / 모델 ---
# ARTISYNTH_HOME: 컴파일된 ArtiSynth 루트(classes/, lib/ 가 있는 곳). JVM classpath 구성에 사용.
ARTISYNTH_HOME = os.environ.get("ARTISYNTH_HOME", "/opt/artisynth/artisynth_core")
# MRI_MODEL: 로드할 ArtiSynth RootModel 클래스. FEM 혀 + TrackingController를 세팅하는 데모.
MRI_MODEL = os.environ.get("MRI_MODEL", "artisynth.models.tongue3d.FemTongueMriDemo")
# MRI_MANIFEST: 모델이 읽는 매니페스트(.properties). 윤곽 CSV, 정합(registration), 가중치 등을 가리킴.
MRI_MANIFEST = os.environ.get(
    "MRI_MANIFEST",
    os.path.join(MRI_FIT_DIR, "mri_fit_tongue.properties"),
)
# TARGETS_CSV: 프레임별 MRI 정중시상 타깃 좌표(모델 metres). 형식: frame,idx,x,y,z
TARGETS_CSV = os.environ.get(
    "TARGETS_CSV",
    os.path.join(MRI_FIT_DIR, "frame_targets_m.csv"),
)
# OUT_CSV: 결과(프레임별 근육 활성도) 저장 경로.
OUT_CSV = os.environ.get(
    "OUT_CSV",
    os.path.join(MRI_OUT, "activations_static_per_frame.csv"),
)
# CONTROLLER: 모델이 만든 TrackingController(역해 컨트롤러)의 이름. 못 찾으면 타입으로 fallback.
CONTROLLER = os.environ.get("CONTROLLER", "mriTracking")

# --- 시뮬레이션 / 램프 ---
# SETTLE_T: 한 프레임당 시뮬레이션 시간 예산(초). 내부적으로 ramp(0.75배)+hold(0.5배)로 나눠 씀.
#           클수록 안정적이지만 프레임당 스텝 수가 늘어 느려짐.
SETTLE_T = float(os.environ.get("SETTLE_T", "0.6"))
# FPS: MRI 촬영 프레임레이트. 결과 CSV의 time 열(=(frame-1)/FPS) 계산에만 사용(동역학 아님).
FPS = float(os.environ.get("FPS", "5.0"))
# NRAMP: 타깃을 rest→목표로 옮길 때 나누는 단계 수. 클수록 천천히 끌어 element 뒤집힘(blow-up) 감소.
NRAMP = int(os.environ.get("NRAMP", "30"))
# MAX_FRAMES: 처리할 최대 프레임 수(앞에서부터). 0이면 전체. 디버그/속도 제한용.
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))
# JVM_XMX: JVM 최대 힙 크기(-Xmx).
JVM_XMX = os.environ.get("JVM_XMX", "4g")
# MAXSTEP: FEM 적분 최대 스텝 크기(초). 작을수록 안정적이지만 스텝 수↑ → 느려짐(가장 큰 속도 인자).
MAXSTEP = float(os.environ.get("MAXSTEP", "0.001"))
# INCOMP: FEM 비압축성 방식(OFF/AUTO/...). 역해 부하에서 AUTO는 자주 터지므로 기본 OFF.
INCOMP = os.environ.get("INCOMP", "OFF").upper()
# MAX_EXCITATION_JUMP: 한 스텝에서 근육 활성도 변화 상한. 작을수록 부드럽지만 수렴 느림.
MAX_EXCITATION_JUMP = float(os.environ.get("MAX_EXCITATION_JUMP", "0.02"))
# INDEPENDENT_FRAMES: True면 프레임마다 rest로 reset(독립 정적해). False면 이어서 풂(연속, 시간 연속성↑).
INDEPENDENT_FRAMES = os.environ.get("INDEPENDENT_FRAMES", "1").lower() not in ("0", "false", "no")
# CHECKPOINT_EVERY: N프레임마다 중간 결과 저장. 0이면 끔.
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "10"))

# --- 3D 표면 피팅(B안) 옵션 ---
# FIT_MODE: surface3d = FEM 표면 노드 다수를 3D(상대변위 RBF)로 맞춤(권장).
#           midsag2d = 기존 방식, dorsal 11개를 MRI 윤곽 절대좌표로 맞춤(과소결정·blow-up 위험).
FIT_MODE = os.environ.get("FIT_MODE", "surface3d").lower()   # surface3d | midsag2d
# N_TARGET_NODES: 타깃으로 쓸 FEM 표면 노드 총 개수. 많을수록 3D 제약↑(정확)·계산량↑(느림).
N_TARGET_NODES = int(os.environ.get("N_TARGET_NODES", "60"))
# RBF_LEN_M: 대칭 RBF 스키닝의 길이 척도(metres). 클수록 변위가 넓고 부드럽게 퍼짐(뻣뻣).
RBF_LEN_M = float(os.environ.get("RBF_LEN_M", "0.018"))
# RBF_SMOOTH: RBF 평활화 항. 클수록 제어점을 정확히 통과하지 않고 부드럽게 근사.
RBF_SMOOTH = float(os.environ.get("RBF_SMOOTH", "1e-6"))
# SUBW: 타깃별 축 가중치 (x,y,z). 측면(y)은 측정값이 아니라 합성이므로 필요시 낮춰도 됨.
#       예) SUBW=1,0.3,1 → 좌우 제약을 약하게.
SUBW = tuple(float(x) for x in os.environ.get("SUBW", "1,1,1").split(","))
# REST_FRAME: 변위 기준이 되는 rest 프레임 id. 0이면 가장 작은 프레임 id를 자동 사용.
REST_FRAME = int(os.environ.get("REST_FRAME", "0"))

# 한 번 init() 하면 채워지는 전역 핸들 캐시(여러 프레임에서 재사용).
_S = {
    "main": None,       # ArtiSynth Main (시뮬레이션 드라이버)
    "root": None,       # RootModel
    "ctrl": None,       # TrackingController (역해 컨트롤러)
    "tpts": None,       # 타깃 점 리스트(컨트롤러의 getTargetPoints)
    "exciters": None,   # 근육 exciter(활성도) 객체 리스트
    "names": None,      # exciter 이름 리스트(결과 CSV 열 순서)
    "tongue": None,     # FEM 혀 모델(FemMuscleModel)
    "rest_pos": None,   # {target_idx: (x,y,z)} 각 타깃 노드의 FEM rest 위치(최초 1회 캡처)
    "n_targets": None,  # 실제 사용하는 타깃 개수
    "node_numbers": None,  # 타깃 idx -> FEM 노드 번호 (forward 비교에서 매칭용)
}


def _log(msg):
    """진행 로그 출력(즉시 flush)."""
    print(msg, flush=True)


def _start_jvm():
    """JVM을 (한 번만) 띄우고 ArtiSynth classes + lib/*.jar 를 classpath에 건다.
    headless 모드라 GUI 없이 동작."""
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
    """Java 클래스 핸들을 얻는 단축 함수(jpype.JClass)."""
    import jpype
    return jpype.JClass(name)


def _find_tongue(root):
    """RootModel 트리를 재귀 탐색해 muscle exciter를 가진 FEM 혀 모델을 찾아 반환."""
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
    """이름(CONTROLLER)으로 TrackingController를 찾고, 실패하면 타입으로 첫 컨트롤러를 반환."""
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
    """모델이 만든 입력 프로브(시계열 타깃 주입)를 모두 끔.
    우리는 프레임별로 타깃 위치를 직접 setPosition 하므로 프로브가 간섭하지 않게 함."""
    try:
        ips = root.getInputProbes()
        for i in range(ips.size()):
            try:
                ips.get(i).setActive(False)
            except Exception:
                pass
    except Exception:
        pass


def _mean_target_error_mm(tpts, n, use_y=False):
    """타깃점(목표)과 그 source 노드(현재 FEM 위치)의 평균 거리(mm).
    use_y=False면 정중시상(x,z)만, True면 3D(x,y,z) 오차를 계산."""
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
        dy = (sp.y - tp_pos.y) if use_y else 0.0
        total += (dx * dx + dy * dy + dz * dz) ** 0.5
        count += 1
    return (total / count * 1000.0) if count else float("nan")


def _fit_error_mm(root, tpts, n):
    """현재 프레임의 피팅 오차(mm). 작을수록 FEM이 타깃 형상에 가깝다는 뜻.
    surface3d는 타깃이 모델 내장 11개를 초과하므로(내장 getTongueFitError는 일부만 커버)
    전체 타깃에 대해 직접 평균을 계산한다."""
    if FIT_MODE == "surface3d":
        return _mean_target_error_mm(tpts, n, use_y=(SUBW[1] != 0.0))
    try:
        return float(root.getTongueFitError()) * 1000.0
    except Exception:
        return _mean_target_error_mm(tpts, n)


def _farthest_point_sample(points, k, seed=0):
    """공간적으로 고르게 퍼진 k개 점의 인덱스를 탐욕적으로 선택(Farthest-Point Sampling).
    표면 노드를 좌우·등배·앞뒤로 골고루 타깃에 넣어 3D 제약이 한쪽에 쏠리지 않게 함."""
    import numpy as np
    pts = np.asarray(points, dtype=float)
    if k >= len(pts):
        return list(range(len(pts)))
    sel = [seed]
    d = np.linalg.norm(pts - pts[seed], axis=1)
    for _ in range(1, k):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return sel


def _setup_surface_targets(ctrl, tongue):
    """[surface3d 전용] 역해 컨트롤러의 타깃을 FEM 표면 노드 N_TARGET_NODES개로 확장.

    절차:
      1) 이미 타깃인 노드(모델 기본 dorsal 11개) 번호를 모아 중복 방지.
      2) 동적(dynamic)이고 표면(surface)인 후보 노드를 수집(고정/내부 노드 제외).
      3) FPS로 고르게 골라 ctrl.addPointTarget()로 타깃에 추가.
      4) 모든 타깃의 축 가중치를 SUBW로 설정(3D 매칭).
    기본 11개는 그대로 두고 run 루프에서 동일한 변위로 함께 구동됨.
    반환값: 최종 타깃 개수."""
    V3 = _jclass("maspack.matrix.Vector3d")
    tpts = ctrl.getTargetPoints()
    existing = set()
    for i in range(tpts.size()):
        src = tpts.get(i).getSourceComp()
        if src is not None:
            existing.add(int(src.getNumber()))

    cand_nodes, cand_pos = [], []
    for i in range(tongue.numNodes()):
        nd = tongue.getNode(i)
        try:
            if not tongue.isSurfaceNode(nd) or not nd.isDynamic():
                continue
        except Exception:
            continue
        if int(nd.getNumber()) in existing:
            continue
        p = nd.getPosition()
        cand_nodes.append(nd)
        cand_pos.append((p.x, p.y, p.z))

    n_add = max(0, N_TARGET_NODES - tpts.size())
    if n_add and cand_nodes:
        for j in _farthest_point_sample(cand_pos, n_add):
            ctrl.addPointTarget(cand_nodes[j], 1.0)

    tpts = ctrl.getTargetPoints()
    for i in range(tpts.size()):
        tpts.get(i).setSubWeights(V3(SUBW[0], SUBW[1], SUBW[2]))
    _log("[init] surface3d: %d surface targets (subWeights=%s), %d candidates"
         % (tpts.size(), SUBW, len(cand_nodes) + len(existing)))
    return tpts.size()


def _capture_rest(tpts, n, m):
    """각 타깃 source 노드의 FEM rest 위치를 1회 캡처해 {idx:(x,y,z)}로 반환.
    rest는 프레임이 바뀌어도 동일하므로(매 프레임 reset) 최초 한 번만 잡아 재사용.
    surface3d에서 '목표 = rest + 변위'의 기준점으로 쓰임."""
    m.reset()
    rest = {}
    for i in range(n):
        src = tpts.get(i).getSourceComp()
        if src is not None:
            p = src.getPosition()
            rest[i] = (p.x, p.y, p.z)
    return rest


def _surface_goals(frames, fr, rest_frame_id, rest_pos, n):
    """[surface3d 전용] 프레임 fr의 각 타깃 3D 목표 위치 {idx:(x,y,z)}를 만든다.

    핵심 아이디어(절대좌표 X, 상대변위 O):
      - MRI 정중시상 제어점의 '변위' = (프레임 fr) - (rest 프레임)  ... (x,z)만
      - 이 변위장을 대칭 가우시안 RBF로 보간해 FEM 표면 노드 위치(x,z)에서 평가
      - 목표 = FEM rest + 보간된 변위(dx, dz),  y는 rest 유지(좌우는 측정값이 아님)
    rest 프레임에서 변위가 0이라 시작 오차가 없어 element 뒤집힘(blow-up)을 회피.
    """
    import numpy as np
    from scipy.interpolate import RBFInterpolator

    f0 = frames[rest_frame_id]      # rest 프레임의 제어점들
    fk = frames[fr]                 # 현재 프레임의 제어점들
    idxs = sorted(set(f0) & set(fk))   # 두 프레임에 공통으로 존재하는 제어점 인덱스
    # 제어점의 rest 위치(x,z)와 변위(dx,dz)
    ctrl_rest = np.array([[f0[i][0], f0[i][2]] for i in idxs])     # (M,2) x,z
    delta = np.array([[fk[i][0] - f0[i][0], fk[i][2] - f0[i][2]] for i in idxs])
    # 변위장 보간기: 제어점 위치 -> 변위. (4_retarget의 대칭 RBF 스키닝과 동일한 발상)
    rbf = RBFInterpolator(
        ctrl_rest, delta, kernel="gaussian",
        epsilon=1.0 / RBF_LEN_M, degree=-1, smoothing=RBF_SMOOTH)

    # 각 FEM 타깃 노드의 rest (x,z)에서 변위를 평가해 목표 위치 산출
    qs = np.array([[rest_pos[i][0], rest_pos[i][2]] for i in range(n)])
    d = rbf(qs)
    goals = {}
    for i in range(n):
        rx, ry, rz = rest_pos[i]
        goals[i] = (rx + d[i, 0], ry, rz + d[i, 1])   # y(=ry)는 rest 유지
    return goals


def init(manifest=None, model=None):
    """JVM 기동 → MRI 모델 로드 → 역해(TrackingController) 준비까지 1회 세팅.

    하는 일:
      1) JVM 시작, ArtiSynth Main 확보, 매니페스트로 모델(loadModel) 로드.
      2) FEM 혀 모델과 TrackingController 핸들 확보.
      3) FEM 안정화 설정: 중력 0, MAXSTEP, 비압축성(INCOMP), 활성도 점프 상한 등.
      4) exciter(근육) 목록/이름 수집.
      5) FIT_MODE=surface3d면 표면 노드 타깃 확장(_setup_surface_targets) + rest 캡처.
    결과 핸들들은 전역 _S 에 저장해 이후 프레임에서 재사용. exciter 이름 리스트 반환."""
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

    rest_pos = None
    if FIT_MODE == "surface3d":
        # 표면 노드 다수를 3D 타깃으로 확장하고 rest 위치를 캡처
        n_targets = _setup_surface_targets(ctrl, tongue)
        tpts = ctrl.getTargetPoints()
        rest_pos = _capture_rest(tpts, n_targets, m)
    else:
        # 기존 2D: 모델 기본 dorsal 타깃 11개만 사용
        tpts = ctrl.getTargetPoints()
        n_targets = min(tpts.size(), 11)

    # 타깃 idx -> FEM 노드 번호 (forward 시뮬에서 같은 노드를 찾아 비교하기 위함)
    node_numbers = []
    for i in range(n_targets):
        src = tpts.get(i).getSourceComp()
        node_numbers.append(int(src.getNumber()) if src is not None else -1)

    _S.update(
        main=m, root=root, ctrl=ctrl, tpts=tpts, exciters=exciters, names=names,
        tongue=tongue, rest_pos=rest_pos, n_targets=n_targets,
        node_numbers=node_numbers,
    )
    mode = "independent (reset/frame)" if INDEPENDENT_FRAMES else "continuous (no reset)"
    _log(
        "[init] ready in %.1fs — %d targets, %d exciters (FIT_MODE=%s)"
        % (time.time() - t0, n_targets, len(exciters), FIT_MODE)
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
    """결과 저장: 헤더 [frame, time, <근육이름들>], 행마다 프레임별 활성도.
    time = (frame-1)/fps. records = [(frame, {name: activation}), ...]."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time"] + names)
        for fr, acts in records:
            t = (fr - 1) / fps
            wr.writerow([fr, "%.4f" % t] + ["%.5f" % acts.get(nm, 0.0) for nm in names])


def _rest_targets(tpts, n):
    """현재(reset 직후) 각 타깃 source 노드 위치를 {idx:(x,y,z)}로 반환.
    램프의 출발점(rest)으로 사용."""
    rest = {}
    for i in range(n):
        tp = tpts.get(i)
        src = tp.getSourceComp()
        if src is not None:
            p = src.getPosition()
            rest[i] = (p.x, p.y, p.z)
    return rest


def _set_targets_blend(tpts, rest, goals, frac, n):
    """타깃 위치를 rest→goal 사이 frac(0~1) 지점으로 선형 보간해 설정.
    frac을 0→1로 키우며 타깃을 천천히 끌어 급격한 변형(blow-up)을 막는다."""
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
    """타깃을 NRAMP 단계로 rest→goal까지 천천히 옮기며 시뮬레이션(ramp), 이후 hold.
    각 단계에서 시뮬 예외(inverted element 등) 발생 시 즉시 실패 처리.
    반환: (성공 여부, 소요 시간 초)."""
    nramp = max(1, NRAMP)
    ramp_t = SETTLE_T * 0.75    # 타깃을 끌어가는 구간 시간
    hold_t = SETTLE_T * 0.5     # 목표 도달 후 안정화(hold) 구간 시간
    seg = ramp_t / nramp        # 단계당 시뮬 시간
    rest = _rest_targets(tpts, n)

    _log("  %sramp targets %d steps (%.2fs) + hold %.2fs..." % (
        ("%s — " % label) if label else "", nramp, ramp_t, hold_t,
    ))
    t0 = time.time()
    ok = True

    for k in range(1, nramp + 1):
        frac = float(k) / nramp
        _set_targets_blend(tpts, rest, goals, frac, n)
        # play(time)은 '현재시각 + time'까지 도는 지속시간(Scheduler.play).
        # 각 단계는 seg 만큼만 전진해야 총 램프 시간이 ramp_t가 됨.
        # (seg*k는 단계마다 누적 재시뮬 → 총 ramp_t*(nramp+1)/2 로 ~6배 낭비. 최종
        #  goal에서 수렴한 값만 기록되므로 정확도 영향 없이 step만 절약된다.)
        m.playAndWait(seg)
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
    """한 프레임의 역해를 풀어 근육 활성도를 얻는다.

    절차: (필요시 reset) → 프로브 끄기 → 타깃을 램프로 끌며 시뮬 → 활성도/오차 기록.
    인자:
      targets : {target_idx:(x,y,z)} 이번 프레임의 목표 위치.
      settle  : 시뮬 시간 예산(None이면 SETTLE_T).
      reset   : True면 rest로 초기화 후 풂(None이면 INDEPENDENT_FRAMES 따름).
    반환: (활성도 dict{name:value}, 피팅오차 mm, 성공 여부 bool)."""
    if _S["main"] is None:
        init()
    if settle is None:
        settle = SETTLE_T
    if reset is None:
        reset = INDEPENDENT_FRAMES

    m, root, tpts, exciters = _S["main"], _S["root"], _S["tpts"], _S["exciters"]
    n = _S["n_targets"] or min(tpts.size(), 11)
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
    """전체 배치 실행: 타깃 CSV의 모든 프레임을 순회하며 역해를 풀고 결과를 저장.

    프레임마다:
      - FIT_MODE에 따라 목표(goals) 생성(surface3d=상대변위 RBF, midsag2d=절대좌표)
      - solve_frame으로 활성도 계산, records에 누적
      - CHECKPOINT_EVERY 마다 중간 저장
    마지막에 OUT_CSV로 전체 저장하고 경로를 반환."""
    if _S["main"] is None:
        init()

    print_paths()
    targets_csv = targets_csv or TARGETS_CSV
    out_csv = out_csv or OUT_CSV # OUT_CSV = /work/mri_fit/activations_static_per_frame.csv
    settle = SETTLE_T if settle is None else settle
    fps = FPS if fps is None else fps # FPS = 5.0
    max_frames = MAX_FRAMES if max_frames is None else max_frames # MAX_FRAMES = 0

    frames = load_targets(targets_csv)
    frame_ids = sorted(frames.keys())
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]

    names = _S["names"]
    total = len(frame_ids)
    rest_frame_id = REST_FRAME if REST_FRAME in frames else frame_ids[0]
    _log("Loaded %d frames from %s" % (total, targets_csv))
    _log("Output -> %s" % out_csv)
    if FIT_MODE == "surface3d":
        _log("surface3d: rest frame=%d, RBF_LEN=%.3fm, %d targets"
             % (rest_frame_id, RBF_LEN_M, _S["n_targets"]))

    records = []
    goal_records = []   # (frame_id, [(x,y,z), ...]) — forward 비교용 타깃 목표 위치
    n = _S["n_targets"] or min(_S["tpts"].size(), 11)
    t_run = time.time()

    for k, fr in enumerate(frame_ids, 1):
        pct = 100.0 * k / total
        t_mri = (fr - 1) / fps
        eta = ""
        if k > 1:
            avg = (time.time() - t_run) / (k - 1)
            eta = " | ETA ~%.0fs" % (avg * (total - k + 1))
        _log("[%d/%d] %.0f%% frame=%d t=%.2fs%s" % (k, total, pct, fr, t_mri, eta))

        # 독립 모드는 매 프레임, 연속 모드는 첫 프레임에서만 rest로 초기화
        do_reset = INDEPENDENT_FRAMES or (k == 1)
        if FIT_MODE == "surface3d":
            # 상대변위를 RBF로 표면 노드에 입힌 3D 목표
            goals = _surface_goals(frames, fr, rest_frame_id, _S["rest_pos"], n)
        else:
            # 기존 2D: CSV의 절대 좌표를 dorsal 타깃에 그대로 대응
            goals = {i: frames[fr][i] for i in range(n) if i in frames[fr]}
        acts, fit_mm, ok = solve_frame(
            goals, settle=settle, frame_no=fr, reset=do_reset,
        )
        if not ok and not INDEPENDENT_FRAMES:
            _log("  -> mesh corrupted; use INDEPENDENT_FRAMES=1 or fix params")
        records.append((fr, acts))
        if FIT_MODE == "surface3d":
            goal_records.append((fr, [goals[i] for i in range(n)]))

        top = sorted(acts.items(), key=lambda kv: -kv[1])[:3]
        _log("  top: " + ", ".join("%s=%.3f" % (nm, v) for nm, v in top))

        if CHECKPOINT_EVERY > 0 and k % CHECKPOINT_EVERY == 0:
            ckpt = out_csv.replace(".csv", "_checkpoint.csv")
            _write_csv(ckpt, names, records, fps)
            _log("[checkpoint] %d frames -> %s" % (k, ckpt))

    _write_csv(out_csv, names, records, fps)
    if FIT_MODE == "surface3d" and goal_records:
        _write_goals_npz(out_csv, goal_records)
    _log(
        "DONE. %d frames in %.1fs -> %s"
        % (len(records), time.time() - t_run, out_csv)
    )
    return out_csv


def _write_goals_npz(out_csv, goal_records):
    """forward 비교용 타깃 목표 위치를 .npz로 저장.
    저장: frame_ids (T,), node_numbers (n,), goals (T, n, 3) — 모두 모델 metres.
    goals[t,i] = 프레임 t에서 타깃 i(FEM 노드 node_numbers[i])가 도달해야 할 3D 위치."""
    import numpy as np
    path = out_csv.replace(".csv", "_goals.npz")
    frame_ids = np.array([fr for fr, _ in goal_records], dtype=int)
    goals = np.array([g for _, g in goal_records], dtype=float)  # (T, n, 3)
    node_numbers = np.array(_S["node_numbers"], dtype=int)
    np.savez(path, frame_ids=frame_ids, node_numbers=node_numbers, goals=goals)
    _log("[out] goals -> %s  (%d frames x %d nodes)"
         % (path, goals.shape[0], goals.shape[1]))


def muscle_names():
    """근육 exciter 이름 목록 반환(필요하면 init 먼저 수행). 결과 CSV 열 순서와 동일."""
    if _S["names"] is None:
        init()
    return list(_S["names"])


def shutdown():
    """JVM 종료 및 전역 상태 초기화(라이브러리로 import해 쓸 때의 정리용)."""
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
        sys.stdout.flush()
        os._exit(130)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    # ArtiSynth keeps non-daemon threads alive; jpype.shutdownJVM() can hang,
    # so exit hard once results are written.
    sys.stdout.flush()
    os._exit(0)
