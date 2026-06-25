#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
artisynth_forward.py

Embed ArtiSynth's FEM-muscle tongue *in-process* in Python via JPype, and expose

    muscle_power(activations) -> (verts, faces)

i.e. set muscle excitations -> the real ArtiSynth solver moves the FEM tongue to
equilibrium -> return the deformed surface mesh. No socket / no separate GUI: the
ArtiSynth Java API (model + muscle exciters + solver) is loaded into this Python
process.

REQUIREMENTS
  * Java (JDK) + the compiled ArtiSynth tree (artisynth_core/classes + lib/*.jar)
  * pip install JPype1
  * Run on the same machine that has ArtiSynth+Java (NOT the slim Python Docker).

CONFIG (env vars)
  ARTISYNTH_HOME  default C:\\Users\\d11\\artisynth\\artisynth_core
  TONGUE_MODEL    default artisynth.models.tongue3d.HexTongueDemo
  SETTLE_T        default 0.4   (seconds of forward sim per call to reach equilibrium)
  JVM_XMX         default 4g

USAGE
  from artisynth_forward import init, muscle_names, muscle_power, save_obj, shutdown
  names = init()                                  # starts JVM, builds the model
  verts, faces = muscle_power([0.3] + [0.0]*(len(names)-1))   # activations 0..1
  save_obj(verts, faces, "pose.obj")
  shutdown()
"""
import os
import glob
import numpy as np

ARTISYNTH_HOME = os.environ.get("ARTISYNTH_HOME", "/opt/artisynth/artisynth_core")
TONGUE_MODEL   = os.environ.get("TONGUE_MODEL", "artisynth.models.tongue3d.HexTongueDemo")
SETTLE_T       = float(os.environ.get("SETTLE_T", "0.4"))
JVM_XMX        = os.environ.get("JVM_XMX", "4g")
# MAXSTEP: FEM 적분 스텝(초). inverse(6번)와 동일하게 맞춰야 같은 변형이 재현됨.
MAXSTEP        = float(os.environ.get("MAXSTEP", "0.001"))   # solver max step (s); small = stable
NRAMP          = int(os.environ.get("NRAMP", "20"))          # ramp activation in N steps (avoids element inversion)
INCOMP         = os.environ.get("INCOMP", "OFF").upper()

_S = {"main": None, "tongue": None, "exciters": None, "names": None,
      "mesh": None, "faces": None}


def _start_jvm():
    import jpype
    if jpype.isJVMStarted():
        return
    cp = [os.path.join(ARTISYNTH_HOME, "classes")] + \
         glob.glob(os.path.join(ARTISYNTH_HOME, "lib", "*.jar"))
    libdir = os.path.join(ARTISYNTH_HOME, "lib")
    jpype.startJVM(
        "-Xmx%s" % JVM_XMX,
        "-Djava.awt.headless=true",
        "-Dartisynth.home=%s" % ARTISYNTH_HOME,
        "-Djava.library.path=%s" % libdir,
        classpath=cp,
    )


def _find_tongue(root):
    """Recursively find the FemMuscleModel that has muscle exciters."""
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


def _jclass(name):
    import jpype
    return jpype.JClass(name)


def init(model=None):
    """Start the JVM, build the model, cache tongue / exciters / mesh. Returns muscle names."""
    import jpype

    _start_jvm()
    JString = _jclass("java.lang.String")
    JArray = jpype.JArray
    Main = _jclass("artisynth.core.driver.Main")
    ArrayList = _jclass("java.util.ArrayList")

    model = model or TONGUE_MODEL
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

    if not m.loadModel(model, model.split(".")[-1], JArray(JString)([])):
        raise RuntimeError("loadModel failed: " + str(m.getErrorMessage()))
    root = m.getRootModel()
    tongue = _find_tongue(root)
    if tongue is None:
        raise RuntimeError("no FemMuscleModel with exciters found in " + model)

    try:
        root.models().get(0).setGravity(0, 0, 0)
        tongue.setGravity(0, 0, 0)
    except Exception:
        pass
    try:
        root.setMaxStepSize(MAXSTEP)      # small step -> stable FEM integration
    except Exception:
        pass
    # STABILITY: HexTongueDemo defaults to IncompMethod.AUTO, whose incompressibility
    # solve can invert elements under load (detJ<0). StableFemMuscleTongueDemo uses
    # OFF -> do the same here. Configurable via INCOMP env (OFF/AUTO/ELEMENT/NODAL).
    try:
        FemModel = _jclass("artisynth.core.femmodels.FemModel")
        mode = os.environ.get("INCOMP", "OFF").upper()
        tongue.setIncompressible(getattr(FemModel.IncompMethod, mode))
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
    _S.update(main=m, tongue=tongue, exciters=exciters, names=names,
              mesh=mesh, faces=_extract_faces(mesh))
    print("ArtiSynth ready: %d exciters, %d surface verts, %d FEM nodes. order: %s"
          % (len(exciters), mesh.numVertices(), tongue.numNodes(), ",".join(names)))
    return list(names)


def _extract_faces(mesh):
    """표면 메쉬 face 인덱스 (F,3) — rest에서 한 번만 추출(토폴로지는 불변)."""
    faces = mesh.getFaces()
    nf = faces.size()
    F = np.empty((nf, 3), dtype=int)
    for i in range(nf):
        vi = faces.get(i).getVertexIndices()
        F[i, 0] = vi[0]; F[i, 1] = vi[1]; F[i, 2] = vi[2]
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


def muscle_names():
    if _S["names"] is None:
        init()
    return list(_S["names"])


def _read_mesh():
    """현재 표면 메쉬 정점 위치 (N,3) + face (F,3). 단위 metres."""
    mesh = _S["mesh"]
    verts = mesh.getVertices()
    nv = verts.size()
    out = np.empty((nv, 3))
    for i in range(nv):
        p = verts.get(i).getPosition()
        out[i, 0] = p.x; out[i, 1] = p.y; out[i, 2] = p.z
    return out, _S["faces"]


def read_surface():
    """현재 표면 메쉬 (verts (N,3), faces (F,3)). metres."""
    return _read_mesh()


def read_nodes():
    """현재 FEM 전체 노드 위치 (Nn,3) 와 노드 번호 (Nn,). 단위 metres.
    노드 '번호'는 inverse(6번)가 저장한 타깃 node_numbers와 매칭하는 키."""
    tongue = _S["tongue"]
    nn = tongue.numNodes()
    pos = np.empty((nn, 3))
    nums = np.empty((nn,), dtype=int)
    nodes = tongue.getNodes()
    for i in range(nn):
        nd = nodes.get(i)
        p = nd.getPosition()
        pos[i, 0] = p.x; pos[i, 1] = p.y; pos[i, 2] = p.z
        nums[i] = int(nd.getNumber())
    return pos, nums


def apply_activation(a, settle=None):
    """활성도를 open-loop로 가해 평형까지 시뮬. 성공 여부(bool) 반환.

    매 호출: rest로 reset → 활성도를 NRAMP 단계로 서서히 올림(급가하면 element 뒤집힘)
    → 최종값으로 hold. 실제 ArtiSynth forward 솔버 사용. 이후 read_nodes/read_surface로
    원하는 위치를 읽으면 됨."""
    if _S["main"] is None:
        init()
    if settle is None:
        settle = SETTLE_T
    if isinstance(a, dict):
        a = [float(a.get(n, 0.0)) for n in _S["names"]]
    m = _S["main"]
    exciters = _S["exciters"]
    m.reset()
    _deactivate_probes(m.getRootModel())
    # play(time)은 지속시간(현재시각+time). 각 단계는 seg만큼만 전진해야 총 램프가 settle.
    # (seg*k는 단계마다 누적 재시뮬 → ~NRAMP/2 배 낭비. 최종 평형값만 쓰므로 정확도 무관.)
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
            print("  -> try lower activation, larger NRAMP, or smaller MAXSTEP")
            ok = False
            break
    if ok:
        m.playAndWait(float(settle) * 2.0)     # hold full activation to settle
        if m.getSimulationException() is not None:
            ok = False
    return ok


def muscle_power(a, settle=None):
    """activations -> (verts (N,3), faces (F,3)). apply_activation 후 표면 메쉬 반환."""
    apply_activation(a, settle=settle)
    return _read_mesh()


def save_obj(verts, faces, path):
    with open(path, "w") as f:
        for v in verts:
            f.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        if faces is not None:
            for t in faces:
                f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))


def shutdown():
    import jpype
    if jpype.isJVMStarted():
        jpype.shutdownJVM()
    _S.update(main=None, tongue=None, exciters=None, names=None, mesh=None)


if __name__ == "__main__":
    names = init()
    print("muscles:", names)
    v, f = muscle_power([0.3] + [0.0] * (len(names) - 1))
    print("verts:", v.shape, "faces:", f.shape)
    save_obj(v, f, "pose_test.obj")
    print("wrote pose_test.obj")
