#!/bin/bash
# One-time setup: if the Linux native solver libs are missing from the mounted
# ArtiSynth tree, fetch them (CRLF-safe: calls the Java installer directly,
# not the bin/updateArtisynthLibs shell script which has Windows line endings).
set -e
AH="${ARTISYNTH_HOME:-/opt/artisynth/artisynth_core}"
NATIVE="$AH/lib/Linux64"

if [ ! -d "$AH" ]; then
  echo "[setup] WARNING: ARTISYNTH_HOME ($AH) not found. Mount your ArtiSynth tree to /opt/artisynth."
elif [ ! -d "$NATIVE" ] || [ -z "$(ls -A "$NATIVE" 2>/dev/null)" ]; then
  echo "[setup] Linux native libs missing -> fetching from artisynth.org (one-time)..."
  if ( cd "$AH" && java -cp "lib/vfs2.jar:bin/libraryInstaller.jar" \
         artisynth.core.driver.LibraryInstaller -updateLibs \
         -remoteSource https://www.artisynth.org/files/lib/ ); then
    echo "[setup] native libs ready in $NATIVE"
  else
    echo "[setup] WARNING: lib fetch failed (need network + read-write mount); native solver may be unavailable"
  fi
else
  echo "[setup] Linux native libs present ($NATIVE)"
fi

exec "$@"
