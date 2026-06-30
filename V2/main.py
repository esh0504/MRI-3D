#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2/main.py

artisynth.utils(fem / vis) 스모크 테스트 하네스.

두 가지 테스트를 제공한다:

  1) test_vis()   - JVM/Java 불필요. rest OBJ로 vis() 3D 렌더 검증 + PNG 저장.

  2) test_fem()   - 실제 ArtiSynth + Java + JPype 필요. fem()으로 모델을 로드하고
                    11D 근육 활성값을 가해 forward 구동한 뒤 vis()로 마스크를 만든다.
                    환경이 없으면(JPype 미설치/Java 없음/모델 경로 오류) 깔끔하게
                    SKIP 처리하고 이유를 출력한다.

실행:
  python main.py            # 사용 가능한 모든 테스트 실행(없으면 SKIP)
  python main.py vis        # 합성 vis 테스트만
  python main.py fem        # 실제 ArtiSynth forward+vis 테스트만
  python main.py fem --activation 0.3,0,0,0,0,0,0,0,0,0,0
  python main.py vis --elev 45 --azim 90   # 카메라 각도 (해부학 좌표, utils.vis와 동일)

카메라 기본값은 아래 VIS_ELEV / VIS_AZIM 상수로도 바꿀 수 있다.
  +Z=위, -Y=우, -X=앞. azim=90=우측 sagittal, azim=0=정면.

test_vis는 실제 혀 rest 메쉬(../tongue_model/tongue_rest_m.obj)를 로드해 렌더한다
(없으면 합성 더미로 폴백). REST_OBJ 환경변수로 경로 변경 가능.

출력 이미지는 V2/_test_out/ 에 저장된다.
  vis_rest.png / vis3d_rest.png   (3D mesh, trimesh+pyrender)
  headless: apt install xvfb libxrender1 && xvfb-run -a python main.py vis
"""
import os
import sys

import numpy as np

# main.py를 V2 밖에서 실행해도 artisynth 패키지를 찾도록 경로 보강
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from artisynth.utils import (  # noqa: E402
    fem, vis, vis3d, MUSCLE_NAMES, TongueModel, model_from_obj,
)

OUT_DIR = os.path.join(HERE, "_test_out")
# vis() 카메라 (해부학 좌표: +Z=위, -Y=우, -X=앞)
RENDER_SETTINGS = dict(upper_degree=45, right_degree=90, size=(640, 640))
# 11D 근육 활성값 0..1 (MUSCLE_NAMES 순). dict {이름: 값} 또는 길이 11 리스트.
ACTIVATIONS = {
    "GGP": 0.0,
    "GGM": 0.0,
    "GGA": 0.0,
    "STY": 0.0,
    "GH": 0.0,
    "MH": 0.0,
    "HG": 0.3,
    "VERT": 0.0,
    "TRANS": 0.0,
    "IL": 0.0,
    "SL": 0.0,
}
# 실제 ArtiSynth 혀 rest 메쉬(OBJ). V2의 형제 폴더 tongue_model/ 에 있음.
REST_OBJ = os.environ.get(
    "REST_OBJ", os.path.join(HERE, "..", "tongue_model", "tongue_rest_m.obj"))


def _save_png(img, name):
    """(H,W,3) uint8 -> PNG 저장. imageio 없으면 조용히 건너뜀."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, img)
        return path
    except Exception as e:
        print("   (PNG 저장 건너뜀: %s)" % e)
        return None


def _synthetic_tongue():
    """간단한 정중시상 평면 메쉬(혀 모양 비슷한 다각형)를 TongueModel로 래핑.

    JVM 없이 vis()를 테스트하기 위한 더미 모델. x-z 평면(midsag)에서 채워진
    실루엣이 나오도록 y=0 근방의 삼각형 팬을 만든다. 단위는 metres 가정."""
    # 혀 단면 비슷한 외곽선 (x:ant-post, z:up), y~0
    outline = np.array([
        [0.00, 0.0, 0.00], [0.02, 0.0, 0.015], [0.04, 0.0, 0.022],
        [0.06, 0.0, 0.020], [0.08, 0.0, 0.012], [0.09, 0.0, 0.000],
        [0.07, 0.0, -0.008], [0.04, 0.0, -0.010], [0.01, 0.0, -0.006],
    ], dtype=float)
    centroid = outline.mean(axis=0, keepdims=True)
    verts = np.vstack([centroid, outline])          # 0=center, 1..N=outline
    n = len(outline)
    faces = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)], dtype=int)

    m = TongueModel()
    m.verts = verts
    m.faces = faces
    m.names = list(MUSCLE_NAMES)
    m.activation = np.zeros(len(MUSCLE_NAMES))
    return m


def _tongue_model():
    """실제 ArtiSynth 혀 rest 메쉬(OBJ)를 로드해 TongueModel로 반환.

    OBJ가 없으면 합성 더미로 폴백 → 어떤 환경에서도 테스트가 돌아간다."""
    if os.path.isfile(REST_OBJ):
        m = model_from_obj(REST_OBJ)
        print("   rest mesh: %s (V=%d, F=%d)"
              % (os.path.relpath(REST_OBJ, HERE), len(m.verts), len(m.faces)))
        return m
    print("   (REST_OBJ 없음 → 합성 더미 사용: %s)" % REST_OBJ)
    return _synthetic_tongue()


def _parse_activation(spec):
    """'0.3,0,...'(11개) 또는 'GGP=0.3,HG=0.2' 형식 -> fem에 넘길 값."""
    if spec is None:
        return [0.3] + [0.0] * 10            # 기본: GGP만 약하게
    if "=" in spec:
        d = {}
        for tok in spec.split(","):
            k, v = tok.split("=")
            d[k.strip()] = float(v)
        return d
    return [float(x) for x in spec.split(",")]



def main(argv):
    args = list(argv)
    activation_spec = None
    if "--activation" in args:
        i = args.index("--activation")
        if i + 1 < len(args):
            activation_spec = args[i + 1]

    print("muscles(11D):", ", ".join(MUSCLE_NAMES), "\n")

    if activation_spec:
        activations = _parse_activation(activation_spec)
    else:
        activations = ACTIVATIONS
    print("activations:", activations)

    try:
        import jpype  # noqa: F401
    except Exception as e:
        print("SKIP fem: JPype 없음 (%s)" % e)
        print("pip install JPype1 후 ARTISYNTH_HOME/Java 설정 필요")
        return

    try:
        model = fem(None, activations)
    except Exception as e:
        print("fem 실패:", e)
        print("ARTISYNTH_HOME / Java / 컴파일된 ArtiSynth 모델 확인")
        return

    print("solver_ok:", getattr(model, "ok", None))
    print("verts:", model.verts.shape, "faces:", model.faces.shape)
    for name, val in zip(MUSCLE_NAMES, model.activation):
        if val > 0:
            print("  %s=%.3f" % (name, val))

    img = vis(model, RENDER_SETTINGS)
    path = _save_png(img, "vis_fem.png")
    if path:
        print("저장:", path)


if __name__ == "__main__":
    main(sys.argv[1:])
