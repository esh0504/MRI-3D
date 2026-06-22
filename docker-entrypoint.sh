#!/bin/bash
# 컨테이너 시작 시 ArtiSynth 준비 상태 확인 (이미지 빌드 시 설치됨).
set -e
AH="${ARTISYNTH_HOME:-/opt/artisynth/artisynth_core}"
NATIVE="$AH/lib/Linux64"

if [ ! -d "$AH/classes" ]; then
  echo "[setup] ERROR: ArtiSynth classes missing at $AH"
  exit 1
fi

if [ ! -d "$NATIVE" ] || [ -z "$(ls -A "$NATIVE" 2>/dev/null)" ]; then
  echo "[setup] Linux native libs missing -> fetching (one-time fallback)..."
  if ( cd "$AH" && java -cp "lib/vfs2.jar:bin/libraryInstaller.jar" \
         artisynth.core.driver.LibraryInstaller -updateLibs \
         -remoteSource https://www.artisynth.org/files/lib/ ); then
    echo "[setup] native libs ready in $NATIVE"
  else
    echo "[setup] ERROR: native lib fetch failed (need network)"
    exit 1
  fi
else
  echo "[setup] ArtiSynth ready: $AH ($(ls "$NATIVE" | wc -l) native libs)"
fi

# FemTongueMriDemo 컴파일 확인
if [ ! -f "$AH/classes/artisynth/models/tongue3d/FemTongueMriDemo.class" ]; then
  echo "[setup] WARNING: FemTongueMriDemo.class not found — inverse may fail"
fi

exec "$@"
