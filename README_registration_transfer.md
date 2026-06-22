# Registration + Transfer — status & how to run

This continues the RT-MRI tongue → ArtiSynth pipeline. Two things were added:
**(1) registration** of the 2D masks into the ArtiSynth model frame, and
**(2) transfer** — both a kinematic 3D lift (runs in Python now) and the wiring for
the muscle-activation inverse (runs in ArtiSynth).

## Key finding: the inverse is already implemented in your tree

`C:\Users\d11\artisynth\...\models\jawTongue\JawFemMuscleTongueMriDemo.java` is a
complete "MRI fitting prototype using a midsagittal contour/landmark observation
model". It already:

- loads a `mri_fit.properties` manifest + `contours.csv` / `landmarks.csv` /
  `registration.csv`;
- fits the image→model registration (`MriRegistration2d`, a 2D affine, needs ≥3
  pairs) and applies it to every frame;
- drives tongue FEM node targets from the resampled tongue contour with
  `subWeights=(1,0,1)` — i.e. tracks x (ant-post) and z (up), **ignores y
  (lateral)**, which *is* the left-right symmetry assumption from the design doc;
- adds the model's bilateral-symmetric `MuscleExciter`s (`GGP_L`+`GGP_R`→`GGP`,
  etc., built automatically) plus jaw/hyoid frame targets;
- writes `*_computed_excitations.txt` (**= muscle activations**) and
  `*_tracked_positions.txt` (**= 3D motion**), with fit-error probes.

So the §7 Java skeleton in the handoff doc is **superseded** — no new driver is
needed. What was missing was the data in the exact format it consumes, which
`export_mri_fit.py` now produces.

## Model frame (verified from the sources)

- Composite Jaw+FEM-tongue model is in **millimetres** (`m2mm = 1000`).
- Midsagittal plane is **y = 0**; **x** = anterior→posterior, **z** = superior(up).
- Tongue geometry = `tongue.obj × 1000`, then shifted **+2 mm in x**.
  Midsagittal tongue landmarks (mm) used as registration anchors:
  tip `(60.39, 99.52)`, dorsum-apex `(100.18, 110.85)`, root `(132.75, 67.32)` `[x,z]`.

## Registration anchor choice (deviation from the doc, with evidence)

The doc proposed palate(2)+teeth(6) as static anchors. Measured centroid drift
over the 105 frames says otherwise:

| label | max centroid drift (px) |
|---|---|
| head (1) | **0.9**  ← most static |
| jaw (3) | 6.9 |
| teeth (6) | 7.8 |
| soft palate (2) | 8.5 (it is articulatory) |
| tongue (4) | 11.1 |
| airway (5) | 16.5 |

So **head(1)** is the truly rigid reference for *frame stabilization*. For the
*image→model* registration itself, the current `registration.csv` uses three
tongue-surface extremes (tip/dorsum/root) at the rest frame matched to the known
model tongue landmarks — a standard subject→model initialization. Refine by
editing `registration.csv` (e.g. click better points in the ArtiSynth viewer, or
add maxilla/incisor pairs) if the fit needs tightening.

Implied scale from these anchors: **≈ 1.16 mm/px** — a data-driven replacement for
the `MM_PER_PX = 1.0` placeholder (the model tongue's physical size fixes it).

## Files produced

```
mri_fit/
  contours.csv        frame,structure,x,y     tongue (40 pts, tip→root) + jaw + palate
  landmarks.csv       frame,label,x,y         jaw centroid per frame
  registration.csv    label,imageX,imageY,modelX,modelZ   tip/dorsum/root anchors
  mri_fit.properties  manifest (frameRate, weights, regularization, csv names)
tongue_lift_3d.npy    (105,25,15,3) kinematic symmetric 3D surface trajectories (mm)
lift_frames3d.png     3 representative lifted frames
lift_motion.gif       lifted surface deforming over the sequence
```

Image coordinate convention in all CSVs: `x = col`, `y = (255 − row)` (up = +y).
Registration verified in Python: anchors reproduce to ~1e-13 mm and the mapped
contour lands inside/near the model tongue bounds.

## Precise contour (updated)

Contour extraction now uses sub-pixel marching-squares (`tongue_contour.py`,
`skimage.find_contours` at 0.5) and keeps the longest airway-facing arc, instead
of a pixel-band + geodesic walk. It hugs the tongue surface (tip→root) far more
faithfully — no corner-cutting or blocky steps. All of `mri_to_artisynth_targets.py`,
`export_mri_fit.py`, the retarget and the comparison GIF were regenerated with it.

## Muscle activations — per frame / per interval

Muscle activation is the **output of the ArtiSynth inverse** (a per-step QP solve
over the FEM model); it cannot be computed without running the FEM solver, so it
is produced by the ArtiSynth run, not in Python. The flow:

1. Run the inverse (below) → it writes `subject1_computed_excitations.txt`
   (one row per time step: `t  a0 a1 … aM`, each `a_i` = activation 0–1 of muscle
   exciter i, in `tongue.getMuscleExciters()` order: GGP, GGM, GGA, STY, GH, MH,
   HG, VERT, TRANS, IL, SL …).
2. Summarize it:

   ```
   python summarize_activations.py subject1_computed_excitations.txt --fps 5 --segments 7
   # or per phone/interval:  --intervals intervals.csv   (columns: start,stop,label)
   ```

   → `activations_per_frame.csv`, `activations_per_interval.csv` (mean+peak per
   muscle per interval), `activations_heatmap.png` (muscles × time), and
   `activations_peaks.png`. (`activations_*.png` currently shown are a SYNTHETIC
   format preview; the CSVs are placeholders until you run the real inverse.)

## Run the activation inverse (in ArtiSynth)

1. Copy the `mri_fit/` folder somewhere on disk (or leave it here).
2. Build ArtiSynth (your `artisynth_core`) as usual.
3. Launch:

   ```
   artisynth -model artisynth.models.jawTongue.JawFemMuscleTongueMriDemo \
             -Dartisynth.mriManifest=C:\Users\d11\Project\Tongue_Inverse\mri_fit\mri_fit.properties
   ```

   (or put `mri_fit.properties` in the ArtiSynth working dir and pass it as the
   model arg.)
4. Press play. Outputs land in the working dir:
   `subject1_computed_excitations.txt` (activations per muscle exciter over time),
   `subject1_tracked_positions.txt` (3D node motion), and `*_fit_error.txt`.

## Tongue-only variant (no jaw) — `FemTongueMriDemo`

A standalone tongue driver was added at
`artisynth_core/src/artisynth/models/tongue3d/FemTongueMriDemo.java`
(registered as `FemTongueMRI` in `mainModels.txt`). It extends `HexTongueDemo`
(the FEM muscle tongue, **metres**), reuses the same manifest / contour / 2D
registration classes, drives the dorsal tongue nodes from the MRI contour
(subWeights (1,0,1)), and uses **Linear** target interpolation (smoother than the
jaw demo's Step → less blow-up at 5 fps). No jaw/hyoid.

It uses its own metre-unit inputs (the image-space `contours.csv` is reused):

```
mri_fit/registration_m.csv        anchors in METRES (tongue.obj coords)
mri_fit/mri_fit_tongue.properties  tongue-only manifest (l2=0.2, damping=0.2, maxJump=0.05)
```

Build + run:

```
cd C:\Users\d11\artisynth\artisynth_core
bin\compile.bat                                   # compile the new .java
bin\artisynth.bat -model artisynth.models.tongue3d.FemTongueMriDemo [ C:\Users\d11\Project\Tongue_Inverse\mri_fit\mri_fit_tongue.properties ]
```

(or Load from class `artisynth.models.tongue3d.FemTongueMriDemo` with Build args =
the `mri_fit_tongue.properties` path). Play → `File → Save output probe data` →
`subject1_computed_excitations.txt`, then `summarize_activations.py` as before.

Note: the standalone tongue has no jaw/hyoid base attachment, so if it is *less*
stable than the jaw version, raise `l2Regularization`/`dampingRegularization` in
the panel. The jaw version with `jawTargetWeight=0` remains the more anchored option.

## Step-by-step in the ArtiSynth GUI

The model is registered as **`JawFemMuscleTongueMRI`** in `mainModels.txt`
(class `artisynth.models.jawTongue.JawFemMuscleTongueMriDemo`).

**A. Launch with the manifest (recommended — one line):**

```
cd C:\Users\d11\artisynth\artisynth_core
bin\artisynth.bat -model artisynth.models.jawTongue.JawFemMuscleTongueMriDemo [ C:\Users\d11\Project\Tongue_Inverse\mri_fit\mri_fit.properties ]
```

The square brackets are ArtiSynth's model-argument syntax; the path is passed to
the model as `args[0]`, which it uses as the manifest. The GUI opens with the
model built and the inverse already configured.

**B. Or pure GUI (no command-line args):**

1. `bin\artisynth.bat` to open the GUI.
2. **File → Set working folder…** → choose `C:\Users\d11\Project\Tongue_Inverse\mri_fit`
   (so the model finds `mri_fit.properties` by default). Do this *before* loading.
3. **Models → Load from class …** → type
   `artisynth.models.jawTongue.JawFemMuscleTongueMriDemo` → OK.
   (If it's also added to your demo menu via `-demoFile mainModels.txt`, it shows
   up under **Models** as *JawFemMuscleTongueMRI*.)

**Once loaded (either way):**

4. The inverse auto-creates a control panel (`mriTracking`) and, in the Timeline,
   the probes: target positions (input), **computed excitations** (output =
   activations), tracked/source positions, and fit-error probes. The tongue is
   colour-mapped by excitation.
5. Set the play stop-time to **21 s** (sequence length) and press **▶ Play**. The
   QP inverse solves each step — the tongue tracks the MRI contour and muscles
   activate. Watch `meanFitError` in the panel to gauge tracking quality.
6. **File → Save output probe data** → writes the attached files into the working
   (manifest) folder, including `subject1_computed_excitations.txt`,
   `subject1_tracked_positions.txt`, `subject1_*_fit_error.txt`.
7. Get per-frame / per-interval activations:

   ```
   python summarize_activations.py C:\Users\d11\Project\Tongue_Inverse\mri_fit\subject1_computed_excitations.txt --fps 5 --segments 7
   ```

**Gotchas**

- The model must be **compiled** first. If *Load from class* throws
  ClassNotFound, build ArtiSynth (`bin\compile.bat`, or your IDE) — your tree has
  the sources under `artisynth_core/src`.
- If the run is unstable on the first frames, lower `maxExcitationJump` or raise
  `l2Regularization`/`dampingRegularization` in `mri_fit.properties` (re-load the
  model after editing). `setMaxStepSize(0.001)` is already set for stability.
- Tracking weights (`tongueTargetWeight`, `jawTargetWeight`) also live in the
  manifest; the inverse control panel lets you toggle/scale terms live.

## To tune / verify next

- `frameRate` is set to the **confirmed 5 fps** (sequence duration = 105/5 = 21 s).
- Tighten registration: refine `registration.csv` anchors or add maxilla points.
- Tracking weights / regularization live in the manifest (`tongueTargetWeight`,
  `l2Regularization`, `dampingRegularization`, `maxExcitationJump`).
- Optionally add an MRI image overlay: set `frameImageDir` to a folder of
  `frame1.png…` and the demo shows the slices behind the model.
- The kinematic lift's lateral profile (`HALF_W`, `EDGE_DROP`) is a heuristic for
  sanity-checking only; the physically correct out-of-plane shape comes from the
  ArtiSynth FEM run.

## Retargeting onto the actual ArtiSynth tongue mesh (`retarget_to_artisynth.py`)

This takes the MRI motion all the way onto the **real ArtiSynth tongue surface
mesh** (`tongue3d/geometry/tongue.obj`, 433 verts / 724 faces), in pure Python —
the kinematic counterpart to the activation inverse.

Method:

1. Load `tongue.obj`, convert to model mm (`×1000`, `+2 mm` x).
2. Apply the image→model affine (from `registration.csv`) to each frame's MRI
   tongue contour.
3. Transfer **relative** motion (vs the rest frame), so the model keeps its own
   rest shape and only the MRI *motion* is retargeted (true motion transfer, not
   shape replacement).
4. Drive the model's dorsal midsagittal curve, then propagate the (x,z)
   displacement to every vertex with a **left-right symmetric Gaussian-RBF
   skinning** that decays away from the dorsum — so the tongue base stays ~fixed
   and lateral (y) shape is preserved.

Outputs:

```
retargeted_tongue.npy   (105, 433, 3) deformed model-tongue vertices (mm)
retarget_midsag.png     model dorsum tracking the MRI contour (3 frames)
retarget_frames3d.png   deformed ArtiSynth tongue surface (3 frames)
retarget_motion.gif     the mesh deforming over the whole sequence
retargeted_objs/        frame_000/026/052/078/104.obj (loadable in ArtiSynth)
```

Smoothing (added): the per-frame control displacement is smoothed **spatially**
along the curve (`SPATIAL_WIN`) and **temporally** across frames (Savitzky-Golay,
`TEMPORAL_WIN`), then re-anchored so the rest frame is undeformed; `RBF_LEN` was
also raised to 18 mm for a stiffer skin. Effect: frame-to-frame jitter dropped to
≈0.6 mm and the posterior-root spike is gone (peak displacement 48→17 mm, mean
per-frame max ≈9 mm) — still consistent with speech tongue excursion.

Frame rate: **5 fps** (user-confirmed) — the time axis (`t = k/5 s`, total 21 s)
and `retarget_motion.gif` now play at real time. `RBF_LEN`, `NCTRL`,
`SPATIAL_WIN`, `TEMPORAL_WIN` are the main tunables. This kinematic retarget needs
no solver; the dynamic, muscle-driven retarget is the ArtiSynth run above (same
registration + contour, manifest `frameRate=5`).
