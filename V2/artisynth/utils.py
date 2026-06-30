#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2/artisynth/utils.py

Analysis-by-Synthesis 파이프라인(SKILL.md)의 ArtiSynth 인터페이스 유틸.
JPype로 ArtiSynth FEM 혀 모델을 *인프로세스* 로드(기존 artisynth_forward.py 방식과
동일)하고, 11D 근육 활성값을 가해 평형까지 forward 구동한 뒤, 정중시상
(midsagittal) 실루엣 마스크 이미지를 반환한다.

핵심 API (요청 스펙):

    model = fem(model, muscle_values)   # model 로드+활성도 적용 → model 반환
    img   = vis(model)                  # (H, W, 3) uint8 3D mesh (trimesh+pyrender)
    path  = vis(model.verts, model.faces, out_path)  # → out_path + ".png" 저장
    rgb   = vis3d(model)                # vis()와 동일 backend (별칭)

사용 예:

    from artisynth.utils import fem, vis, MUSCLE_NAMES

    model = None
    model = fem(model, [0.3] + [0.0] * 10)   # 11D 활성값 (MUSCLE_NAMES 순서)
    mask  = vis(model)                        # (H, W, 3) uint8
    # 두 번째 프레임: model 핸들 재사용(JVM/모델 로드 1회만)
    model = fem(model, {"GGP": 0.5, "HG": 0.2})

환경 변수 (기본값):
    ARTISYNTH_HOME   ArtiSynth 트리 경로 (classes + lib/*.jar)
    TONGUE_MODEL     기본 artisynth.models.tongue3d.HexTongueDemo
                     (커스텀 컴파일 시 ...FemTongueMriDemo 사용 가능)
    SETTLE_T  (0.4)  forward 평형까지 시뮬 시간(초)
    MAXSTEP   (0.001) FEM 적분 스텝(초)
    NRAMP     (20)   활성도 램프 단계 수(급가하면 element 뒤집힘)
    INCOMP    (OFF)  비압축성 방법(OFF/AUTO/ELEMENT/NODAL)
    JVM_XMX   (4g)   JVM 힙

요구 사항:
    * Java(JDK) + 컴파일된 ArtiSynth 트리
    * pip install JPype1
    * vis() / vis3d(): trimesh + pyrender off-screen 3D (pip install trimesh pyrender)
    * headless Linux: apt install xvfb libxrender1 libx11-6
      DISPLAY 없으면 PyVirtualDisplay로 xvfb 자동 기동 (pip install PyVirtualDisplay)
    * vis_mask(): 2D midsagittal 실루엣 (최적화 loss용, matplotlib)
"""
import glob
import os

import numpy as np

# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #
ARTISYNTH_HOME = os.environ.get(
    "ARTISYNTH_HOME", r"C:\Users\d11\artisynth\artisynth_core")
TONGUE_MODEL = os.environ.get(
    "TONGUE_MODEL", "artisynth.models.tongue3d.HexTongueDemo")
SETTLE_T = float(os.environ.get("SETTLE_T", "0.4"))
MAXSTEP = float(os.environ.get("MAXSTEP", "0.001"))
NRAMP = int(os.environ.get("NRAMP", "20"))
INCOMP = os.environ.get("INCOMP", "OFF").upper()
JVM_XMX = os.environ.get("JVM_XMX", "4g")

# 11D 제어 공간(SKILL.md). inverse(6번)/activations CSV 헤더 순서와 동일.
MUSCLE_NAMES = ["GGP", "GGM", "GGA", "STY", "GH", "MH",
                "HG", "VERT", "TRANS", "IL", "SL"]


# --------------------------------------------------------------------------- #
# Model 핸들
# --------------------------------------------------------------------------- #
class TongueModel:
    """ArtiSynth FEM 혀 모델 상태 핸들.

    JPype 핸들(main/tongue/exciters)과 토폴로지(faces), 그리고 마지막으로 적용한
    활성값에 대한 변형 표면 정점(verts)을 담는다. fem()이 채워서 반환한다.
    """

    def __init__(self):
        self.main = None          # artisynth.core.driver.Main
        self.tongue = None        # FemMuscleModel
        self.exciters = None      # list[MuscleExciter] (model 순서)
        self.names = None         # list[str] 모델이 보고한 exciter 이름(순서)
        self.mesh = None          # 표면 메쉬 핸들
        self.faces = None         # (F,3) int  표면 삼각형 인덱스(불변)
        self.verts = None         # (N,3) float  현재 표면 정점(metres)
        self.activation = None    # (11,) float  마지막 적용 활성값

    @property
    def loaded(self):
        return self.main is not None


# --------------------------------------------------------------------------- #
# OBJ 로더 (JVM 불필요) — rest 형상 시각화/테스트용
# --------------------------------------------------------------------------- #
def load_obj(path):
    """Wavefront OBJ → (verts (N,3) float, faces (F,3) int).

    다각형 face는 fan 삼각분할한다. 단위는 파일 그대로(혀 OBJ는 metres)."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in t[1:]]
                for k in range(1, len(idx) - 1):       # fan triangulation
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def model_from_obj(path):
    """OBJ 표면 메쉬를 TongueModel로 래핑(verts/faces만 채움).

    JVM/ArtiSynth 없이 실제 혀 rest 메쉬로 vis()/vis3d()를 검증할 때 사용한다.
    근육 활성값에 의한 *변형*은 여전히 fem()(ArtiSynth)이 필요하다."""
    v, f = load_obj(path)
    m = TongueModel()
    m.verts = v
    m.faces = f
    m.names = list(MUSCLE_NAMES)
    m.activation = np.zeros(len(MUSCLE_NAMES))
    return m


# --------------------------------------------------------------------------- #
# JPype / JVM
# --------------------------------------------------------------------------- #
def _start_jvm():
    import jpype
    if jpype.isJVMStarted():
        return
    cp = ([os.path.join(ARTISYNTH_HOME, "classes")]
          + glob.glob(os.path.join(ARTISYNTH_HOME, "lib", "*.jar")))
    libdir = os.path.join(ARTISYNTH_HOME, "lib")
    jpype.startJVM(
        "-Xmx%s" % JVM_XMX,
        "-Djava.awt.headless=true",
        "-Dartisynth.home=%s" % ARTISYNTH_HOME,
        "-Djava.library.path=%s" % libdir,
        classpath=cp,
    )


def _jclass(name):
    import jpype
    return jpype.JClass(name)


def _find_tongue(root):
    """muscle exciter를 가진 FemMuscleModel을 재귀 탐색."""
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


def _extract_faces(mesh):
    """표면 메쉬 face 인덱스 (F,3). 토폴로지는 불변이라 rest에서 1회만 추출."""
    faces = mesh.getFaces()
    nf = faces.size()
    F = np.empty((nf, 3), dtype=int)
    for i in range(nf):
        vi = faces.get(i).getVertexIndices()
        F[i, 0] = vi[0]
        F[i, 1] = vi[1]
        F[i, 2] = vi[2]
    return F


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


def _read_surface(model):
    """현재 표면 정점 (N,3) metres."""
    mesh = model.mesh
    verts = mesh.getVertices()
    nv = verts.size()
    out = np.empty((nv, 3))
    for i in range(nv):
        p = verts.get(i).getPosition()
        out[i, 0] = p.x
        out[i, 1] = p.y
        out[i, 2] = p.z
    return out


def _load(model_name=None):
    """JVM 시작 + 모델 빌드 → 채워진 TongueModel 반환."""
    import jpype

    _start_jvm()
    JString = _jclass("java.lang.String")
    JArray = jpype.JArray
    Main = _jclass("artisynth.core.driver.Main")
    ArrayList = _jclass("java.util.ArrayList")

    model_name = model_name or TONGUE_MODEL
    m = Main.getMain()
    if m is None:
        try:
            Main.main(JArray(JString)(["-noGui"]))
        except Exception as e:
            print("note: Main.main(-noGui) raised:", e)
        m = Main.getMain()
    if m is None:
        m = Main("forward", False)
        m.start(ArrayList())

    if not m.loadModel(model_name, model_name.split(".")[-1], JArray(JString)([])):
        raise RuntimeError("loadModel failed: " + str(m.getErrorMessage()))
    root = m.getRootModel()
    tongue = _find_tongue(root)
    if tongue is None:
        raise RuntimeError(
            "no FemMuscleModel with exciters found in " + model_name)

    # 중력 제거 (활성값에 의한 변형만 보기 위함)
    try:
        root.models().get(0).setGravity(0, 0, 0)
        tongue.setGravity(0, 0, 0)
    except Exception:
        pass
    try:
        root.setMaxStepSize(MAXSTEP)
    except Exception:
        pass
    # 비압축성 solve가 하중하에서 element를 뒤집을 수 있어 OFF 권장.
    try:
        FemModel = _jclass("artisynth.core.femmodels.FemModel")
        tongue.setIncompressible(getattr(FemModel.IncompMethod, INCOMP))
        try:
            tongue.setMaxStepSize(MAXSTEP)
        except Exception:
            pass
    except Exception as e:
        print("note: setIncompressible failed:", e)
    _deactivate_probes(root)

    exlist = tongue.getMuscleExciters()
    exciters = [exlist.get(i) for i in range(exlist.size())]
    names = [str(e.getName()) for e in exciters]
    mesh = tongue.getSurfaceMesh()

    model = TongueModel()
    model.main = m
    model.tongue = tongue
    model.exciters = exciters
    model.names = names
    model.mesh = mesh
    model.faces = _extract_faces(mesh)
    model.verts = _read_surface(model)
    print("ArtiSynth ready: %d exciters, %d surface verts, %d FEM nodes. order: %s"
          % (len(exciters), mesh.numVertices(), tongue.numNodes(), ",".join(names)))
    return model


# --------------------------------------------------------------------------- #
# 활성값 적용
# --------------------------------------------------------------------------- #
def _coerce_activation(muscle_values, names):
    """muscle_values(list/np/dict) → exciter 순서(names)에 맞춘 (M,) float 벡터."""
    if muscle_values is None:
        return np.zeros(len(names), dtype=float)
    if isinstance(muscle_values, dict):
        return np.array([float(muscle_values.get(n, 0.0)) for n in names],
                        dtype=float)
    a = np.asarray(muscle_values, dtype=float).ravel()
    # 입력 벡터는 MUSCLE_NAMES(SKILL 11D) 순서로 간주한다. 모델이 보고한 exciter
    # 순서(names)가 다르면 이름 기준으로 재배열해 위치를 맞춘다.
    if a.shape[0] == len(MUSCLE_NAMES) and set(names) == set(MUSCLE_NAMES):
        idx = [MUSCLE_NAMES.index(n) for n in names]
        return a[idx]
    if a.shape[0] == len(names):
        # 표준 11D가 아니거나 이름이 다른 모델 → 이미 모델 순서라고 가정.
        return a
    raise ValueError(
        "muscle_values 길이 %d != exciter 수 %d (순서: %s)"
        % (a.shape[0], len(names), ",".join(names)))


def _apply_activation(model, a, settle=None):
    """활성값을 open-loop로 가해 평형까지 forward 시뮬. 성공 여부(bool) 반환.

    매 호출: rest로 reset → NRAMP 단계로 서서히 올림 → 최종값으로 hold.
    실제 ArtiSynth forward 솔버를 사용한다."""
    if settle is None:
        settle = SETTLE_T
    m = model.main
    exciters = model.exciters
    m.reset()
    _deactivate_probes(m.getRootModel())
    seg = float(settle) / NRAMP
    ok = True
    for k in range(1, NRAMP + 1):
        frac = float(k) / NRAMP
        for i, e in enumerate(exciters):
            e.setExcitation((float(a[i]) if i < len(a) else 0.0) * frac)
        m.playAndWait(seg)
        ex = m.getSimulationException()
        if ex is not None:
            print("WARNING: solver exception during ramp step %d: %s" % (k, ex))
            print("  -> 더 낮은 활성도/큰 NRAMP/작은 MAXSTEP 시도")
            ok = False
            break
    if ok:
        m.playAndWait(float(settle) * 2.0)   # 최종값 hold하여 평형
        if m.getSimulationException() is not None:
            ok = False
    return ok


# --------------------------------------------------------------------------- #
# 공개 API: fem
# --------------------------------------------------------------------------- #
def fem(model=None, muscle_values=None, settle=None, model_name=None):
    """ArtiSynth FEM 혀 모델을 로드(필요 시)하고 11D 근육 활성값을 적용한다.

    Parameters
    ----------
    model : TongueModel or None
        이전에 로드한 핸들. None이면 JVM 시작 + 모델 빌드(첫 호출). 이후 프레임에선
        반환된 핸들을 다시 넘겨 재사용(로드 1회만).
    muscle_values : array-like(11,) or dict or None
        근육 활성값 0..1. 길이 11 벡터(MUSCLE_NAMES 순서) 또는 {이름: 값} dict.
        None이면 rest(전부 0).
    settle : float, optional
        평형까지 시뮬 시간(초). 기본 SETTLE_T.
    model_name : str, optional
        ArtiSynth 모델 클래스명. 기본 TONGUE_MODEL.

    Returns
    -------
    TongueModel
        변형된 표면 정점(model.verts, (N,3) metres), faces(model.faces),
        적용 활성값(model.activation, (11,))이 채워진 핸들.
    """
    if model is None or not getattr(model, "loaded", False):
        model = _load(model_name)

    a = _coerce_activation(muscle_values, model.names)
    ok = _apply_activation(model, a, settle=settle)
    model.activation = a
    model.ok = ok
    model.verts = _read_surface(model)
    return model


# --------------------------------------------------------------------------- #
# 공개 API: vis / vis3d  (trimesh + pyrender 3D)
# --------------------------------------------------------------------------- #
def _silhouette_mask(verts, faces, size=(256, 256), bounds=None,
                     axes=(0, 2), margin=0.05, fill=(255, 255, 255),
                     bg=(0, 0, 0)):
    """표면 삼각형을 2D 평면(axes)에 투영하여 채운 실루엣 마스크 (H,W,3) uint8.

    matplotlib(Agg)만 사용 → JVM 불필요. axes=(0,2)는 정중시상(x-z) 평면.
    bounds=(amin,amax,bmin,bmax)를 주면 프레임 간 좌표계를 고정(시계열 비교용).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    H, W = int(size[0]), int(size[1])
    ai, bi = axes
    P = np.asarray(verts, dtype=float)[:, [ai, bi]]   # (N,2): (수평, 수직)
    F = np.asarray(faces, dtype=int)

    if bounds is None:
        amin, amax = P[:, 0].min(), P[:, 0].max()
        bmin, bmax = P[:, 1].min(), P[:, 1].max()
        da = (amax - amin) or 1.0
        db = (bmax - bmin) or 1.0
        amin -= da * margin; amax += da * margin
        bmin -= db * margin; bmax += db * margin
    else:
        amin, amax, bmin, bmax = bounds

    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])           # 여백 0
    ax.set_xlim(amin, amax)
    ax.set_ylim(bmin, bmax)                    # z up: y축 위쪽이 +z
    ax.axis("off")
    ax.set_facecolor(tuple(c / 255.0 for c in bg))
    fig.patch.set_facecolor(tuple(c / 255.0 for c in bg))

    polys = P[F]                               # (F,3,2) 각 삼각형 꼭짓점
    coll = PolyCollection(polys, closed=True,
                          facecolors=[tuple(c / 255.0 for c in fill)],
                          edgecolors="none", antialiaseds=False)
    ax.add_collection(coll)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
    plt.close(fig)
    return buf


def _vis_png_path(out_path):
    """out_path → 저장 경로. 확장자 없으면 .png 붙임."""
    p = os.path.abspath(str(out_path))
    if not p.lower().endswith(".png"):
        p += ".png"
    return p


def _write_vis_png(img, out_path):
    """(H,W,3) uint8 → PNG 저장. 저장된 절대 경로 반환."""
    path = _vis_png_path(out_path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, img)
    except ImportError as e:
        raise ImportError("vis save: pip install imageio") from e
    return path


def _require_mesh(model):
    if (model is None or getattr(model, "verts", None) is None
            or getattr(model, "faces", None) is None):
        raise ValueError("vis: model이 비었습니다. 먼저 fem(...)을 호출하세요.")


# ArtiSynth 혀 mesh 해부학 좌표 (world, metres)
# +Z = superior (위), -Y = patient right (우), -X = anterior (앞)
AX_UP = np.array([0.0, 0.0, 1.0])
AX_RIGHT = np.array([0.0, -1.0, 0.0])
AX_FRONT = np.array([-1.0, 0.0, 0.0])


def _look_at_pose(eye, target, up=None):
    """World-from-camera pose (pyrender convention). up 기본값 = +Z (superior)."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    if up is None:
        up = AX_UP
    up = np.asarray(up, dtype=float)
    z = eye - target
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    if n < 1e-8:
        up = AX_FRONT if abs(np.dot(up, AX_UP)) > 0.9 else AX_UP
        x = np.cross(up, z)
        n = np.linalg.norm(x)
    x /= n + 1e-12
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def _camera_pose_from_angles(center, distance, upper_degree, right_degree):
    """해부학 구면좌표로 eye 위치를 잡고 centroid를 바라본다.

    기준축: +Z=위, -Y=우, -X=앞 (ArtiSynth 혀 mesh convention).

    upper_degree : XY(앞·우) 평면에서 +Z 쪽으로 기울임(deg). 0=수평, 90=위에서 내려봄.
    right_degree : 수평면에서 AZ_FRONT(앞) 기준 CCW(deg).
                   0=정면(앞), 90=우측(-Y), 180=뒤(+X), 270=좌(+Y).

    right_degree=0, upper_degree=0 일 때 화면: 위=+Z, 오른쪽=-Y, 깊이=+X(입안쪽).
    right_degree=90, upper_degree=0 일 때: 정중시상(sagittal) — 위=+Z, 오른쪽=+X(뒤).
    """
    elev = np.deg2rad(float(upper_degree))
    azim = np.deg2rad(float(right_degree))
    c = np.asarray(center, dtype=float)
    ce, se = np.cos(elev), np.sin(elev)
    ca, sa = np.cos(azim), np.sin(azim)
    offset = float(distance) * (ce * ca * AX_FRONT + ce * sa * AX_RIGHT + se * AX_UP)
    return _look_at_pose(c + offset, c, up=AX_UP)


_virtual_display = None


def _ensure_headless_display():
    """DISPLAY 없는 headless 환경에서 pyrender용 가상 X11(xvfb)을 띄운다."""
    global _virtual_display
    if os.environ.get("DISPLAY"):
        return
    if _virtual_display is not None:
        return
    try:
        from pyvirtualdisplay import Display
    except ImportError as e:
        raise RuntimeError(
            "vis: DISPLAY가 없습니다 (headless 환경).\n"
            "  apt install xvfb libxrender1 libx11-6\n"
            "  pip install PyVirtualDisplay\n"
            "또는: xvfb-run -a python main.py"
        ) from e
    _virtual_display = Display(visible=0, size=(1024, 768))
    _virtual_display.start()


def _import_pyrender():
    """pyrender 서브모듈 lazy import (Viewer/pyglet 창은 사용하지 않음)."""
    from pyrender.offscreen import OffscreenRenderer
    from pyrender.scene import Scene
    from pyrender.mesh import Mesh
    from pyrender.material import MetallicRoughnessMaterial
    from pyrender.light import DirectionalLight
    from pyrender.camera import PerspectiveCamera
    return {
        "OffscreenRenderer": OffscreenRenderer,
        "Scene": Scene,
        "Mesh": Mesh,
        "MetallicRoughnessMaterial": MetallicRoughnessMaterial,
        "DirectionalLight": DirectionalLight,
        "PerspectiveCamera": PerspectiveCamera,
    }


def _render_mesh3d_pyrender(verts, faces, size=(768, 768), bg=(28, 28, 36),
                              color=(230, 90, 75), upper_degree=45.0,
                              right_degree=90.0):
    """trimesh + pyrender off-screen 3D → (H, W, 3) uint8 RGB."""
    _ensure_headless_display()
    try:
        import trimesh
        pr = _import_pyrender()
    except ImportError as e:
        raise ImportError(
            "vis: pip install trimesh pyrender (+ headless: apt install xvfb libxrender1)"
        ) from e

    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vis: verts must be (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("vis: faces must be (F, 3)")

    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    rgba = [c / 255.0 for c in color] + [1.0]
    material = pr["MetallicRoughnessMaterial"](
        baseColorFactor=rgba,
        metallicFactor=0.15,
        roughnessFactor=0.55,
    )
    pr_mesh = pr["Mesh"].from_trimesh(tm, material=material, smooth=True)

    scene = pr["Scene"](
        bg_color=[bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0],
        ambient_light=[0.25, 0.25, 0.28],
    )
    scene.add(pr_mesh)

    center = tm.centroid
    dist = float(max(tm.extents.max(), 1e-4) * 2.4)
    cam_pose = _camera_pose_from_angles(center, dist, upper_degree, right_degree)
    camera = pr["PerspectiveCamera"](yfov=np.pi / 4.0, znear=dist * 0.01,
                                     zfar=dist * 20.0)
    scene.add(camera, pose=cam_pose)

    light = pr["DirectionalLight"](color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)
    fill = pr["DirectionalLight"](color=[0.9, 0.9, 1.0], intensity=1.2)
    scene.add(fill, pose=_camera_pose_from_angles(
        center, dist, upper_degree + 15, right_degree + 90))

    H, W = int(size[0]), int(size[1])
    renderer = pr["OffscreenRenderer"](viewport_width=W, viewport_height=H)
    try:
        img, _ = renderer.render(scene)
    finally:
        renderer.delete()

    img = np.asarray(img)[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def vis_mask(model_or_verts, faces=None, size=(256, 256), bounds=None,
             plane="midsag", fill=(255, 255, 255), bg=(0, 0, 0)):
    """2D midsagittal 실루엣 마스크 (matplotlib). 최적화 loss / MRI 비교용."""
    axes = {"midsag": (0, 2), "axial": (0, 1), "coronal": (1, 2)}[plane]
    if faces is not None:
        verts = np.asarray(model_or_verts, dtype=float)
        f = np.asarray(faces, dtype=int)
    else:
        _require_mesh(model_or_verts)
        verts, f = model_or_verts.verts, model_or_verts.faces
    return _silhouette_mask(verts, f, size=size, bounds=bounds,
                            axes=axes, fill=fill, bg=bg)


def vis3d(model, size=(768, 768), bg=(28, 28, 36), color=(230, 90, 75),
          upper_degree=45.0, right_degree=90.0, show_edges=False):
    """vis()와 동일한 trimesh+pyrender 3D 렌더 (show_edges는 현재 미사용)."""
    _require_mesh(model)
    return _render_mesh3d_pyrender(
        model.verts, model.faces, size=size, bg=bg, color=color,
        upper_degree=upper_degree, right_degree=right_degree,
    )


def _apply_vis_settings(settings, defaults):
    """rendering_settings dict → vis() 옵션. upper_degree/right_degree (구 elev/azim 호환)."""
    if not settings:
        return dict(defaults)
    out = dict(defaults)
    for key in ("size", "bg", "color", "out_path", "upper_degree", "right_degree"):
        if key in settings:
            out[key] = settings[key]
    if "elev" in settings:
        out["upper_degree"] = settings["elev"]
    if "azim" in settings:
        out["right_degree"] = settings["azim"]
    return out


def vis(model_or_verts, faces=None, out_path=None, size=(768, 768),
        bg=(28, 28, 36), color=(230, 90, 75), upper_degree=45.0,
        right_degree=90.0, **kwargs):
    """3D mesh를 trimesh+pyrender로 렌더한다.

    호출 형태:

    1) ``vis(model)`` → (H, W, 3) uint8 ndarray
    2) ``vis(model, rendering_settings)`` → settings dict
       (예: ``dict(upper_degree=30, right_degree=90, size=(640, 640))``)
    3) ``vis(model, upper_degree=30, right_degree=90)`` → 키워드 인자
    4) ``vis(verts, faces, out_path, ...)`` → PNG 저장, 경로(str) 반환

    rendering_settings 키: upper_degree, right_degree, size, bg, color, out_path
    (+Z=위, -Y=우, -X=앞. right_degree=90=우측 sagittal)
    """
    defaults = dict(size=size, bg=bg, color=color,
                    upper_degree=upper_degree, right_degree=right_degree,
                    out_path=out_path)

    # vis(model, rendering_settings) — 두 번째 인자가 dict
    if isinstance(faces, dict):
        opts = _apply_vis_settings(faces, defaults)
        opts.update({k: v for k, v in kwargs.items()
                     if k in defaults or k in ("elev", "azim")})
        if "elev" in kwargs:
            opts["upper_degree"] = kwargs["elev"]
        if "azim" in kwargs:
            opts["right_degree"] = kwargs["azim"]
        faces = None
    else:
        opts = _apply_vis_settings(kwargs, defaults)

    size = opts["size"]
    bg = opts["bg"]
    color = opts["color"]
    upper_degree = opts["upper_degree"]
    right_degree = opts["right_degree"]
    out_path = opts["out_path"]

    if out_path is not None:
        if faces is None:
            raise ValueError("vis(verts, faces, out_path): faces가 필요합니다.")
        img = _render_mesh3d_pyrender(
            model_or_verts, faces, size=size, bg=bg, color=color,
            upper_degree=upper_degree, right_degree=right_degree,
        )
        return _write_vis_png(img, out_path)

    model = model_or_verts
    _require_mesh(model)
    return _render_mesh3d_pyrender(
        model.verts, model.faces, size=size, bg=bg, color=color,
        upper_degree=upper_degree, right_degree=right_degree,
    )


# --------------------------------------------------------------------------- #
# 정리
# --------------------------------------------------------------------------- #
def shutdown():
    """JVM 종료(프로세스 당 1회만 시작/종료 가능)."""
    import jpype
    if jpype.isJVMStarted():
        jpype.shutdownJVM()


if __name__ == "__main__":
    # 스모크 테스트: 모델 로드 → rest+활성값 → 마스크 저장
    m = fem(None, [0.3] + [0.0] * 10)
    print("verts:", None if m.verts is None else m.verts.shape,
          "faces:", None if m.faces is None else m.faces.shape,
          "ok:", getattr(m, "ok", None))
    img = vis(m)
    print("vis ->", img.shape, img.dtype, "fg_px:", int((img.sum(-1) > 0).sum()))
    try:
        import imageio.v2 as imageio
        imageio.imwrite("vis_test.png", img)
    except Exception as e:
        print("imwrite skipped:", e)
