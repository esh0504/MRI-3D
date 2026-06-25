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
MRI_FIT_DIR = os.path.join(MRI_OUT, "mri_fit")

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
