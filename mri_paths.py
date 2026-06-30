#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI → ArtiSynth 파이프라인 공통 경로 설정.

기본 레이아웃 (/work = 프로젝트 루트):
  datasets/GT_Segmentations/Subject{1..5}/mask_*.mat   입력 GT
  output/Subject{N}/                                    스크립트 1~5 산출물
  output/Subject{N}/mri_fit/                            스크립트 2 → 6 입력

환경변수 (선택):
  MRI_SUBJECT   Subject1 … Subject5 (또는 1, 2, …, subject3)
  MRI_WORK      프로젝트 루트 (기본: 이 파일이 있는 디렉터리 = /work)
  MRI_ROOT      GT 마스크 폴더 (기본: datasets/GT_Segmentations/<MRI_SUBJECT>)
  MRI_OUT       출력 루트 (기본: output/<MRI_SUBJECT>)
  ARTISYNTH_HOME / TONGUE_OBJ  ArtiSynth tongue.obj 위치
"""
import glob
import os
import re
import sys

WORK_ROOT = os.environ.get(
    "MRI_WORK",
    os.path.dirname(os.path.abspath(__file__)),
)


def normalize_subject(name):
    """'Subject1' | '1' | 'subject2' → 'Subject1' / 'Subject2'."""
    if name is None or str(name).strip() == "":
        return "Subject1"
    name = str(name).strip()
    if re.fullmatch(r"\d+", name):
        return "Subject%d" % int(name)
    m = re.match(r"(?i)subject\s*(\d+)", name)
    if m:
        return "Subject%s" % m.group(1)
    return name


MRI_SUBJECT = normalize_subject(os.environ.get("MRI_SUBJECT", "Subject1"))

GT_BASE = os.path.join(WORK_ROOT, "datasets", "GT_Segmentations")


def _has_masks(path):
    return bool(glob.glob(os.path.join(path, "mask_*.mat")))


def _resolve_mri_root():
    """MRI_ROOT: env 우선. 단, 경로 없거나 mask_*.mat 없으면 datasets/SubjectN 으로 fallback."""
    default = os.path.join(GT_BASE, MRI_SUBJECT)
    env = os.environ.get("MRI_ROOT")
    if not env:
        return default
    if os.path.isdir(env) and _has_masks(env):
        return env
    if env != default:
        print(
            "[paths] WARNING: MRI_ROOT=%s 에 mask_*.mat 없음 → %s 사용"
            % (env, default),
            file=sys.stderr,
        )
    return default


def _resolve_mri_out():
    """MRI_OUT: env 우선. 단, /work 처럼 프로젝트 루트만 가리키면 output/SubjectN 으로 fallback."""
    default = os.path.join(WORK_ROOT, "output", MRI_SUBJECT)
    env = os.environ.get("MRI_OUT")
    if not env:
        return default
    # Docker 기본값 MRI_OUT=/work 은 산출물이 루트에 섞이므로 output/SubjectN 권장
    if os.path.normpath(env) == os.path.normpath(WORK_ROOT):
        print(
            "[paths] NOTE: MRI_OUT=%s (프로젝트 루트) → %s 사용 권장 경로로 대체"
            % (env, default),
            file=sys.stderr,
        )
        return default
    return env


MRI_ROOT = _resolve_mri_root()
MRI_OUT = _resolve_mri_out()
# step 2 의 ArtiSynth 입력 번들 폴더(번호 접두사 포함).
MRI_FIT_DIR = os.path.join(MRI_OUT, "2_mri_fit")


def _sanitize_legacy_env():
    """컨테이너 이미지(Dockerfile)가 박아둔 구(舊) 경로 env 를 무력화.

    OUT_CSV / TARGETS_CSV / MRI_MANIFEST / ACT_CSV 가 '현재 MRI_OUT 바깥'(예: 예전
    /work/mri_fit)을 가리키면 stale 로 보고 제거 → 각 스크립트의 번호 붙은 기본 경로가
    적용된다. MRI_OUT 아래를 가리키는 의도적 지정은 그대로 둔다."""
    base = os.path.abspath(MRI_OUT) + os.sep
    for k in ("OUT_CSV", "TARGETS_CSV", "MRI_MANIFEST", "ACT_CSV"):
        v = os.environ.get(k)
        if v and not os.path.abspath(v).startswith(base):
            os.environ.pop(k, None)
            print("[paths] ignore stale %s=%s (컨테이너 기본값) → 번호 경로 사용" % (k, v),
                  file=sys.stderr)


_sanitize_legacy_env()


def out(n, name):
    """번호 접두사 출력 경로: MRI_OUT/{n}_{name}.
    각 스크립트는 자기 번호로 산출물을 저장해 어느 스텝이 만든 파일인지 드러낸다.
    예) out(1, 'tongue_targets.npy') -> output/SubjectX/1_tongue_targets.npy"""
    return os.path.join(MRI_OUT, "%d_%s" % (n, name))


def clean_all(verbose=True):
    """MRI_OUT 내부를 전부 비운다(폴더 자체는 유지). 전체 파이프라인 시작 시 사용."""
    if not os.path.isdir(MRI_OUT):
        os.makedirs(MRI_OUT, exist_ok=True)
        return
    import shutil
    removed = 0
    for name in os.listdir(MRI_OUT):
        p = os.path.join(MRI_OUT, name)
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            removed += 1
        except Exception as e:
            print("[clean] skip %s (%s)" % (p, e), file=sys.stderr)
    if verbose:
        print("[clean] wiped %d entries under %s" % (removed, MRI_OUT))


def clean_step(n, verbose=True):
    """이 스텝(N_*) 출력만 삭제. 개별 스텝 재실행 시 자기 산출물만 정리."""
    import shutil
    prefix = "%d_" % n
    removed = 0
    if os.path.isdir(MRI_OUT):
        for name in os.listdir(MRI_OUT):
            if not name.startswith(prefix):
                continue
            p = os.path.join(MRI_OUT, name)
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
                removed += 1
            except Exception as e:
                print("[clean] skip %s (%s)" % (p, e), file=sys.stderr)
    if verbose and removed:
        print("[clean] step %d: removed %d previous outputs" % (n, removed))

ARTISYNTH_HOME = os.environ.get("ARTISYNTH_HOME", "/opt/artisynth/artisynth_core")
_default_tongue_obj = os.path.join(
    ARTISYNTH_HOME,
    "src", "artisynth", "models", "tongue3d", "geometry", "tongue.obj",
)
TONGUE_OBJ = os.environ.get("TONGUE_OBJ", _default_tongue_obj)

CLIP_ID = os.environ.get("CLIP_ID", MRI_SUBJECT.lower())


def print_paths():
    """해석된 경로를 stdout에 출력."""
    print("[paths] WORK_ROOT   = %s" % WORK_ROOT)
    print("[paths] MRI_SUBJECT = %s" % MRI_SUBJECT)
    print("[paths] MRI_ROOT    = %s" % MRI_ROOT)
    print("[paths] MRI_OUT     = %s" % MRI_OUT)
    print("[paths] MRI_FIT_DIR = %s" % MRI_FIT_DIR)
    print("[paths] TONGUE_OBJ  = %s" % TONGUE_OBJ)
