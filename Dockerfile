# Python pipeline for RT-MRI -> ArtiSynth tongue (extraction, registration,
# retarget, kinematic lift, activation summary, and the forward-model CLIENT).
#
# This image does NOT contain ArtiSynth/Java/Jython. ArtiSynth (GUI + solver)
# runs on the host; the container's tongue_forward client reaches it over a
# socket at TONGUE_HOST:TONGUE_PORT (default host.docker.internal:5005).
FROM python:3.12-slim

# libgomp1 = OpenMP runtime needed by scipy / scikit-image wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# analysis scripts (Jython files are copied too but are run inside ArtiSynth, not here)
COPY *.py /app/

ENV PYTHONPATH=/app \
    MPLBACKEND=Agg \
    MRI_ROOT=/data/masks \
    MRI_OUT=/work \
    FORWARD_DIR=/work/forward \
    TONGUE_OBJ=/data/artisynth/artisynth_core/src/artisynth/models/tongue3d/geometry/tongue.obj \
    TONGUE_HOST=host.docker.internal \
    TONGUE_PORT=5005

WORKDIR /work
CMD ["bash"]
