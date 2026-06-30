#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_mri_fit.py

Convert the 2D RT-MRI segmentation masks (mask_*.mat, labels 0-6) into the EXACT
input bundle consumed by ArtiSynth's existing class

    artisynth.models.jawTongue.JawFemMuscleTongueMriDemo

which runs the inverse muscle-activation tracking on a midsagittal contour and
writes computed_excitations.txt (= activations) + tracked_positions (= 3D motion).

Outputs (into OUT_DIR/mri_fit/):
    contours.csv       frame,structure,x,y      (ordered surface points, image space)
    landmarks.csv      frame,label,x,y          (jaw anchor per frame, image space)
    registration.csv   label,imageX,imageY,modelX,modelZ   (>=3 static anchors)
    mri_fit.properties Java manifest tying it all together

Image coordinate convention used for ALL csv x,y (must be consistent so the
affine fit in MriRegistration2d maps everything the same way):
    x = col                      (anterior = small x; face points left)
    y = (H-1) - row              (superior = +y, matches model z-up)

Model frame (from artisynth_core sources):
    composite Jaw+FEM-tongue model is in MILLIMETRES (m2mm=1000),
    midsagittal plane y=0, x=anterior-posterior, z=superior-inferior.
    Tongue geometry = tongue.obj * 1000 then +2mm in x. Midsag landmark coords:
        tip (60.39, 99.52)  dorsum-apex (100.18, 110.85)  root (132.75, 67.32)  [x,z mm]

Run JavaScript-free; just numpy/scipy. After running, copy OUT_DIR/mri_fit to the
ArtiSynth working dir and launch:
    artisynth -model artisynth.models.jawTongue.JawFemMuscleTongueMriDemo \
              -Dartisynth.mriManifest=/path/to/mri_fit.properties

경로 (기본 /work 기준):
  GT   datasets/GT_Segmentations/Subject{N}/
  OUT  output/Subject{N}/mri_fit/

환경변수: MRI_SUBJECT=Subject1|2|…|5, MRI_ROOT, MRI_OUT (선택)
"""

import os, re, glob
import numpy as np
import scipy.io as sio
from collections import deque
from scipy.ndimage import binary_dilation, label
from tongue_contour import precise_contour, anatomical_landmarks
from mri_paths import MRI_ROOT, MRI_OUT, MRI_FIT_DIR, CLIP_ID, print_paths

# ----------------------------- CONFIG ---------------------------------------
ROOT       = MRI_ROOT
OUT_DIR    = MRI_OUT
VAR_NAME   = "mask_frame"
LBL = dict(head=1, palate=2, jaw=3, tongue=4, airway=5, teeth=6)
FPS        = 5.0          # actual frame rate (user-confirmed)
REST_FRAME = 1            # 1-based frame used to anchor the static registration
N_TONGUE_CONTOUR = 40    # exported tongue-surface points (ArtiSynth resamples down)
INTERFACE_DILATE = 1

# Model-frame (mm) anchor coords, computed from tongue.obj*1000 (+2mm x). [x, z]
MODEL_ANCHORS = {
    "tip":        (60.39,  99.52),
    "dorsum":     (100.18, 110.85),
    "root":       (132.75, 67.32),
}
# Same anchors in METRES, for the tongue-only model (FemTongueMriDemo / HexTongueDemo,
# which are in metres = tongue.obj raw coords). [x, z]
MODEL_ANCHORS_M = {
    "tip":    (0.05839, 0.09952),
    "dorsum": (0.09818, 0.11085),
    "root":   (0.13075, 0.06732),
}
N_TONGUE_NODES = 11      # = len(DEFAULT_TONGUE_TARGET_NODES) for the static inverse
# Composite (mm) model = tongue.obj*1000 then +X_OFFSET_MM in x. So mm anchors are
# derived from the metres anchors as (x*1000 + X_OFFSET_MM, z*1000). Keep ONE source
# of truth (metres) so registration.csv (mm) and registration_m.csv stay consistent.
X_OFFSET_MM = 2.0
# Known model-frame landmark coords in METRES [x, z] (tongue-only model = obj raw).
# Add more here (or via landmark_map.csv) as you identify corresponding model points.
MODEL_LANDMARKS_M = {
    "tip":    (0.05839, 0.09952),
    "dorsum": (0.09818, 0.11085),
    "root":   (0.13075, 0.06732),
    # "floor": (?, ?),   # fill in if/when you find the model correspondence
}
# ----------------------------------------------------------------------------

_NB8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def natkey(p):
    n = re.findall(r"\d+", os.path.basename(p));  return int(n[-1]) if n else 0

def frames():
    fs = glob.glob(os.path.join(ROOT, "mask_*.mat")); fs.sort(key=natkey); return fs

def load(path):
    d = sio.loadmat(path)
    return d[VAR_NAME] if VAR_NAME in d else next(v for k,v in d.items() if not k.startswith("__"))

def img_xy(rc, H):
    """(row,col) -> (x=col, y=H-1-row)."""
    return np.column_stack([rc[:,1], (H-1) - rc[:,0]])

# ---- tongue<->airway interface, geodesic-ordered tip->root (proven method) ----
def interface_pixels(mask):
    tongue = (mask == LBL["tongue"]); airway = (mask == LBL["airway"])
    if tongue.sum() == 0: return None
    band = (tongue & binary_dilation(airway, iterations=INTERFACE_DILATE)) if airway.sum() else np.zeros_like(tongue)
    if band.sum() < 4: band = binary_dilation(~tongue) & tongue
    lab, n = label(band, structure=np.ones((3,3)))
    if n == 0: return None
    if n > 1:
        s = np.bincount(lab.ravel()); s[0] = 0; band = lab == int(np.argmax(s))
    rc = np.column_stack(np.nonzero(band))
    return rc if len(rc) >= 4 else None

def geodesic_order(rc):
    idx = {(int(r),int(c)):i for i,(r,c) in enumerate(rc)}
    nb = [[] for _ in rc]
    for i,(r,c) in enumerate(rc):
        for dr,dc in _NB8:
            j = idx.get((int(r)+dr,int(c)+dc))
            if j is not None: nb[i].append(j)
    def bfs(s):
        dist=[-1]*len(rc); par=[-1]*len(rc); dist[s]=0; q=deque([s]); last=s
        while q:
            u=q.popleft(); last=u
            for v in nb[u]:
                if dist[v]<0: dist[v]=dist[u]+1; par[v]=u; q.append(v)
        return last,par
    a,_=bfs(0); b,par=bfs(a)
    path=[]; u=b
    while u!=-1: path.append(u); u=par[u]
    return rc[path].astype(float)

def tongue_contour(mask):
    return precise_contour(mask, N_TONGUE_CONTOUR)          # (N,2) row,col tip->root

# ---- ordered closed boundary for jaw / palate (angle sort; for centroid use) ----
def closed_boundary(mask, lbl):
    m = (mask == lbl)
    if m.sum() < 4: return None
    bnd = m & binary_dilation(~m)
    rc = np.column_stack(np.nonzero(bnd)).astype(float)
    c = rc.mean(0); ang = np.arctan2(rc[:,0]-c[0], rc[:,1]-c[1])
    return rc[np.argsort(ang)]

def centroid(mask, lbl):
    rc = np.column_stack(np.nonzero(mask == lbl)).astype(float)
    return None if len(rc)==0 else rc.mean(0)


# ---- image<->model landmark registration (N>=3, least-squares affine) ----------
def model_m_to_mm(xz_m):
    """metres [x,z] -> composite-model mm [x,z] (x gets +X_OFFSET_MM)."""
    return (xz_m[0] * 1000.0 + X_OFFSET_MM, xz_m[1] * 1000.0)


def load_landmark_map():
    """Optional user-defined correspondences. Looked up at MRI_FIT_DIR/landmark_map.csv
    then MRI_OUT/landmark_map.csv. Columns: label,imageX,imageY,modelX_m,modelZ_m
    (model coords in METRES). Rows with blank/NaN model coords are ignored.
    Returns {label:(imageX,imageY,modelX_m,modelZ_m)} or {} if no file."""
    import csv
    for p in (os.path.join(MRI_FIT_DIR, "landmark_map.csv"),
              os.path.join(OUT_DIR, "landmark_map.csv")):
        if not os.path.isfile(p):
            continue
        out = {}
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    out[r["label"].strip()] = (
                        float(r["imageX"]), float(r["imageY"]),
                        float(r["modelX_m"]), float(r["modelZ_m"]))
                except (ValueError, KeyError, TypeError):
                    continue          # blank model coords -> skip this row
        print(f"[reg] using user landmark_map.csv: {p} ({len(out)} usable rows)")
        return out
    return {}


def fit_affine(img_xy, mod_xz):
    """Least-squares affine image[x,y] -> model[x,z]. img_xy (N,2), mod_xz (N,2).
    Returns (A (3,2), rms_residual_in_model_units, per_point_residual (N,))."""
    M = np.column_stack([img_xy, np.ones(len(img_xy))])
    A, *_ = np.linalg.lstsq(M, mod_xz, rcond=None)
    pred = M @ A
    res = np.linalg.norm(pred - mod_xz, axis=1)
    return A, float(np.sqrt((res ** 2).mean())), res


def build_registration(rest_mask, rest_tongue):
    """Assemble image<->model correspondences (metres) for registration.
    Priority: user landmark_map.csv > auto anatomical landmarks (tip/dorsum/root
    [+floor if model coord known]). Falls back to rest_tongue extremes if needed.
    Returns (names, img_xy(N,2), mod_xz_m(N,2))."""
    H = rest_mask.shape[0]
    user = load_landmark_map()
    names, img, mod = [], [], []
    if len(user) >= 3:
        for k, (ix, iy, mx, mz) in user.items():
            names.append(k); img.append([ix, iy]); mod.append([mx, mz])
        return names, np.array(img), np.array(mod)

    # auto: anatomical landmarks from the rest mask, matched to known model coords
    lm = anatomical_landmarks(rest_mask) or {}
    for k in ("tip", "dorsum", "root", "floor"):
        if k in lm and k in MODEL_LANDMARKS_M:
            r, c = lm[k]
            names.append(k); img.append([c, (H - 1) - r]); mod.append(list(MODEL_LANDMARKS_M[k]))
    if len(names) >= 3:
        return names, np.array(img), np.array(mod)

    # last-resort fallback: tongue extremes (old behavior)
    tip = rest_tongue[0]; root = rest_tongue[-1]
    apex = rest_tongue[np.argmax(rest_tongue[:, 1])]
    names = ["tip", "dorsum", "root"]
    img = [tip, apex, root]
    mod = [MODEL_LANDMARKS_M["tip"], MODEL_LANDMARKS_M["dorsum"], MODEL_LANDMARKS_M["root"]]
    return names, np.array(img), np.array(mod)


def write_landmark_template(rest_mask):
    """Write landmark_map_template.csv seeded with rest-frame image coords and known
    model metres so the user can fill in extra correspondences (e.g. floor)."""
    H = rest_mask.shape[0]
    lm = anatomical_landmarks(rest_mask) or {}
    path = os.path.join(MRI_FIT_DIR, "landmark_map_template.csv")
    with open(path, "w") as f:
        f.write("label,imageX,imageY,modelX_m,modelZ_m\n")
        for k in ("tip", "dorsum", "root", "floor"):
            if k not in lm:
                continue
            r, c = lm[k]; ix, iy = c, (H - 1) - r
            mx, mz = MODEL_LANDMARKS_M.get(k, ("", ""))
            f.write(f"{k},{ix:.3f},{iy:.3f},{mx},{mz}\n")
    print(f"[out] {path}  (fill model coords + add rows, save as landmark_map.csv)")


def main():
    print_paths()
    out = MRI_FIT_DIR
    os.makedirs(out, exist_ok=True)
    fs = frames();  assert fs, f"no masks under {ROOT}"
    H = load(fs[0]).shape[0]
    print(f"[info] {len(fs)} frames, H={H}")

    fc = open(os.path.join(out, "contours.csv"), "w");  fc.write("frame,structure,x,y\n")
    fl = open(os.path.join(out, "landmarks.csv"), "w"); fl.write("frame,label,x,y\n")

    rest_tongue = None
    rest_mask = None
    tongues = {}                         # fi -> (N,2) image xy, for frame_targets
    for fi, fp in enumerate(fs, start=1):
        m = load(fp)
        if fi == REST_FRAME: rest_mask = m
        # tongue
        tc = tongue_contour(m)
        if tc is not None:
            xy = img_xy(tc, H)
            tongues[fi] = xy
            for x,y in xy: fc.write(f"{fi},tongue,{x:.3f},{y:.3f}\n")
            if fi == REST_FRAME: rest_tongue = xy
        # jaw + palate boundaries (optional, for overlay / centroid targets)
        for name in ("jaw","palate"):
            b = closed_boundary(m, LBL[name])
            if b is not None:
                for x,y in img_xy(b, H): fc.write(f"{fi},{name},{x:.3f},{y:.3f}\n")
        # jaw landmark = jaw centroid (demo uses relative motion vs rest frame)
        jc = centroid(m, LBL["jaw"])
        if jc is not None:
            x,y = jc[1], (H-1)-jc[0]; fl.write(f"{fi},jaw,{x:.3f},{y:.3f}\n")
    fc.close(); fl.close()
    print(f"[out] {out}/contours.csv\n[out] {out}/landmarks.csv")

    # ---- registration : N>=3 landmark correspondences, least-squares affine ----
    assert rest_tongue is not None, "rest frame tongue contour missing"
    assert rest_mask is not None, "rest frame mask missing"
    write_landmark_template(rest_mask)
    reg_names, reg_img, reg_mod_m = build_registration(rest_mask, rest_tongue)
    # fit affine (image[x,y] -> model[x,z], metres) and report residual
    A_m, rms_m, res_m = fit_affine(reg_img, reg_mod_m)
    print(f"[reg] {len(reg_names)} landmarks {reg_names} | RMS residual "
          f"{rms_m*1000:.2f} mm (worst {res_m.max()*1000:.2f} mm)")
    if len(reg_names) > 3:
        print("[reg] (N>3: over-determined least-squares fit — more robust)")

    # registration.csv (composite model, mm) — derived from metres anchors
    with open(os.path.join(out,"registration.csv"),"w") as f:
        f.write("label,imageX,imageY,modelX,modelZ\n")
        for k, ixy, mxz in zip(reg_names, reg_img, reg_mod_m):
            mx, mz = model_m_to_mm(mxz)
            f.write(f"{k},{ixy[0]:.3f},{ixy[1]:.3f},{mx:.3f},{mz:.3f}\n")
    print(f"[out] {out}/registration.csv  (rest frame {REST_FRAME}, {len(reg_names)} anchors)")

    # ---- manifest ----
    dur = len(fs)/FPS
    props = f"""# ArtiSynth MRI-fit manifest for JawFemMuscleTongueMriDemo
# generated by export_mri_fit.py
clipId={CLIP_ID}
frameRate={FPS}
frameCount={len(fs)}
contourCsv=contours.csv
landmarkCsv=landmarks.csv
registrationCsv=registration.csv
tongueStructure=tongue
jawStructure=jaw
palateStructure=palate
jawAnchorLabel=jaw
# tracking weights / regularization (tune these)
tongueTargetWeight=1.0
jawTargetWeight=0.0
hyoidTargetWeight=0.0
l2Regularization=0.2
dampingRegularization=0.2
maxExcitationJump=0.05
# tongueTargetNodes= (leave blank to use demo default dorsal nodes)
"""
    with open(os.path.join(out,"mri_fit.properties"),"w") as f: f.write(props)
    print(f"[out] {out}/mri_fit.properties  (duration {dur:.2f}s)")

    # sanity: image->model scale implied by the fitted affine (area-based mm/px)
    lin = A_m[:2, :]
    sc = (abs(np.linalg.det(lin)) ** 0.5) * 1000.0
    print(f"[stat] fitted affine scale ~{sc:.3f} mm/px (from {len(reg_names)} landmarks)")

    # ===== tongue-only (metres) inputs: for FemTongueMriDemo + static_inverse =====
    with open(os.path.join(out,"registration_m.csv"),"w") as f:
        f.write("label,imageX,imageY,modelX,modelZ\n")
        for k, ixy, mxz in zip(reg_names, reg_img, reg_mod_m):
            f.write(f"{k},{ixy[0]:.3f},{ixy[1]:.3f},{mxz[0]:.6f},{mxz[1]:.6f}\n")
    print(f"[out] {out}/registration_m.csv  ({len(reg_names)} anchors)")

    tprops = f"""# Tongue-only MRI fit manifest (metres model: FemTongueMriDemo / HexTongueDemo)
clipId={CLIP_ID}
frameRate={FPS}
frameCount={len(fs)}
contourCsv=contours.csv
registrationCsv=registration_m.csv
tongueStructure=tongue
tongueTargetWeight=1.0
l2Regularization=0.2
dampingRegularization=0.2
maxExcitationJump=0.05
"""
    with open(os.path.join(out,"mri_fit_tongue.properties"),"w") as f: f.write(tprops)
    print(f"[out] {out}/mri_fit_tongue.properties")

    # frame_targets_m.csv : per-frame 11 tongue targets in MODEL METRES (for static_inverse)
    # reuse the fitted N-landmark affine A_m (image[x,y] -> model[x,z], metres)
    def to_model_m(xy):
        return np.column_stack([xy, np.ones(len(xy))]) @ A_m
    def resample(c, n):
        d = np.r_[0, np.cumsum(np.hypot(np.diff(c[:,0]), np.diff(c[:,1])))]
        if d[-1] == 0: return np.repeat(c[:1], n, 0)
        u = np.linspace(0, d[-1], n)
        return np.column_stack([np.interp(u,d,c[:,0]), np.interp(u,d,c[:,1])])
    with open(os.path.join(out,"frame_targets_m.csv"),"w") as f:
        f.write("frame,idx,x,y,z\n")
        for fi in sorted(tongues):
            mm = to_model_m(tongues[fi])          # (N,2) [x,z] metres
            s = resample(mm, N_TONGUE_NODES)
            for i in range(N_TONGUE_NODES):
                f.write(f"{fi},{i},{s[i,0]:.6f},0.000000,{s[i,1]:.6f}\n")
    print(f"[out] {out}/frame_targets_m.csv  ({len(tongues)} frames x {N_TONGUE_NODES})")

if __name__ == "__main__":
    main()
