---
name: tongue-mri-inverse
description: >
  Estimate tongue muscle activations from real-time MRI (RT-MRI) of speech by
  retargeting 2D midsagittal tongue motion onto the 3D ArtiSynth biomechanical
  tongue and solving an inverse (muscle-activation) fit, then validating by
  forward simulation. Use for any task in the Tongue_Inverse project: extracting
  tongue contours/landmarks from segmentation masks, image->model registration,
  2D->3D kinematic retargeting, optical-flow point tracking, ArtiSynth static
  inverse / forward, and round-trip comparison. Triggers: RT-MRI tongue,
  midsagittal contour, ArtiSynth, muscle activation/excitation, retarget,
  optical flow, tongue.obj, mask_*.mat, FemTongueMriDemo.
---

# Tongue RT-MRI -> ArtiSynth muscle-activation inverse

## Research goal
From midsagittal RT-MRI of a speaking subject, recover the **tongue muscle
activations** that drive the observed motion, using a physically-based 3D tongue
model (ArtiSynth). Four conceptual stages:

1. **Shape extraction** - 2D MRI frame (binary) -> segment-mask / tongue surface contour / full boundary
   /anatomical landmarks (midsagittal, 2D).
2. **2D->3D retargeting** - transfer the 2D midsagittal motion onto the real 3D
   ArtiSynth tongue mesh (symmetric RBF skinning).
3. **Inverse fit** - in ArtiSynth, solve for muscle excitations whose FEM
   deformation reproduces the target shape (TrackingController).
4. **Validation** - forward-simulate with the predicted activations and compare
   the resulting mesh to the retarget / MRI (round-trip closure).

Stages 1-2 and 3-4 are largely independent problems and can be debugged
separately.

## Pipeline (numbered scripts, run in order)
- `1_extract_contours.py`  masks -> `tongue_targets.npy` (dorsal arc, spur-trimmed),
  `tongue_boundary.npy` (full closed outline), `landmarks_auto.csv`.
- `2_export_artisynth_inputs.py`  -> `mri_fit/` bundle: contours, registration
  (N>=3 least-squares affine, optional user `landmark_map.csv`), manifests,
  `frame_targets_m.csv` (per-frame model-metre targets).
- `3_kinematic_lift.py`  (optional) symmetry-assumption 3D dome lift (sanity).
- `4_retarget_to_artisynth.py`  real tongue.obj deformed to MRI motion via
  symmetric Gaussian-RBF skinning -> `retargeted_tongue.npy`. `TARGETS_NPY` env
  picks the driving target file (e.g. flow-tracked).
- `5_compare_gif.py`  MRI vs retarget side-by-side GIF.
- `6_static_inverse.py`  [ArtiSynth+JPype] per-frame static inverse -> muscle
  activations CSV. Modes: `surface3d` (relative-displacement RBF, robust) /
  `midsag2d`. `INDEPENDENT_FRAMES` resets each frame (default) vs continuous.
- `7_summarize_activations.py`  activations -> tables/heatmaps (expects probe txt;
  the 6b CSV has frame,time,... columns so treat as optional).
- `8_forward_from_activations.py`  [ArtiSynth+JPype] apply activations -> FEM
  forward -> `forward_*.npy`.
- `9_compare_forward.py`  forward vs retarget surface distance (Chamfer) + 3-panel
  MRI/retarget/forward GIF.

Helpers (not numbered): `tongue_contour.py` (precise_contour / full_boundary_contour
/ anatomical_landmarks / posterior-spur clip), `mri_paths.py` (path + `out()`
helper), `artisynth_forward.py` / `forward_server.py` (forward), `parallel_runner.py`
(frame-parallel 6/8), `optical_flow_track.py`, `tongue_flow_arrows.py`,
`color_midsag_vertices.py`, `overlay_mri_vs_model_midsag.py`, `run_all.py`
(orchestrator).

## Run
- Whole pipeline: `MRI_SUBJECT=Subject1 python3 run_all.py`
- Heavy steps in parallel (auto cores/RAM): `python3 run_all.py --parallel`
- Subset: `python3 run_all.py --only 6,8,9` / `--from 4` / `--to 5`
- Flow-driven retarget: set `TARGETS_NPY=tongue_targets_flow.npy` for steps 4/5.

## Key conventions (do not break)
- Image mm: `x = col*MM_PER_PX`, `y = (H-1-row)*MM_PER_PX`, `MM_PER_PX≈1.164`.
- Model frames: tongue-only model is **metres** (raw tongue.obj). Composite
  jaw+tongue is **mm** = `obj*1000` then **+2 mm in x** (`X_OFFSET_MM=2.0`).
  So `mm = m*1000` for z, `m*1000 + 2` for x. Keep registration.csv (mm) and
  registration_m.csv (metres) consistent from one source.
- Frame indexing: masks/`tongue_targets` are 0-based by position; ArtiSynth frame
  ids are 1-based (`retarget_idx = frame_id - 1`).
- Forward (HexTongueDemo) and inverse (FemTongueMriDemo, which *extends*
  HexTongueDemo) share the SAME FEM tongue and node numbering. Keep both at
  gravity=0, INCOMP=OFF, same MAXSTEP so identical excitations give identical shapes.
- Surface meshes differ in tessellation: retarget tongue.obj = 433 verts, FEM
  surface = 370 verts, FEM nodes = 948. Compare by surface-to-surface (Chamfer),
  not vertex-to-vertex; match FEM nodes by node NUMBER.

## Hard-won domain knowledge / gotchas
- **2D correspondence problem**: a midsagittal silhouette point ("highest dorsum")
  is NOT a fixed material point - it changes tissue each frame, and interior
  points are untrackable in plain cine MRI. Two remedies: (a) recover
  correspondence with **optical flow** (advect seeds -> trajectories;
  `optical_flow_track.py`, `direct` mode rest->k ≈ 1.5-2.3 mm vs per-frame,
  `chain` mode drifts), or (b) avoid per-point matching by fitting the model to
  the silhouette (what retarget/inverse already do; model priors fill the rest).
- **Through-plane / internal motion**: single midsag slice cannot see lateral
  motion; interior is regularization, not measurement. True internal material
  motion needs **tagged-MRI (HARP)**; lateral needs 3D/multi-slice. State this
  limit; do not claim lateral muscles (transversus) are identifiable from 2D.
- **2D vs 3D tongue extent mismatch**: the MRI "tongue" label includes a posterior
  pharyngeal/curl-back spur the ArtiSynth model lacks. Trim it with
  `precise_contour(clip_root=True, clip_drop_frac≈1.0)` (cut at x-reversal or when
  z drops ~one dorsum-rise below the peak) so 2D extent matches the model. Clip is
  only stable when combined with flow tracking (per-frame clip point jitters).
- **Inverse continuous-mode blow-up**: with `INDEPENDENT_FRAMES=0`, a single
  inverted-element failure (~mid sequence) corrupts all later frames -> activations
  flatline to 0 -> forward freezes ("moves then stops"). Use independent mode, or
  add reset-on-failure, or soften (NRAMP up, MAXSTEP/MAX_EXCITATION_JUMP down).
- **Validation must target the right thing**: compare forward to the **retarget /
  MRI shape**, not to the inverse's own goals.npz (that only proves closure).
- **Underdetermination**: static inverse is non-unique; denser surface targets +
  L2/effort regularization help. Lateral targets are synthetic (y kept at rest).
- **Registration**: use N>=3 anatomical landmarks (tip/dorsum/root/floor + any
  user-added) with least-squares affine; report RMS residual. Naive dense
  dorsal-arc correspondence by arc-length is unreliable (match by AP/x fraction).

## Data layout
- `datasets/GT_Segmentations/SubjectN/mask_*.mat`  (labels 0-6; tongue=4, airway=5)
- `datasets/MRI_SSFP_10fps/SubjectN/image_*.dcm`    (raw cine, for optical flow)
- `output/SubjectN/...`  pipeline outputs (paths via `mri_paths`).
- tongue mesh: `$ARTISYNTH_HOME/.../tongue3d/geometry/tongue.obj` (env `TONGUE_OBJ`).

## Requirements
Python: numpy, scipy, scikit-image, matplotlib, imageio, pydicom. ArtiSynth steps
(6/8): JDK + compiled ArtiSynth (`ARTISYNTH_HOME`) + JPype1. FPS=5 (user-confirmed).
