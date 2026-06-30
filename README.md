# RT-MRI 혀 → ArtiSynth 파이프라인 (실행 순서)

MRI 분할 마스크에서 혀 컨투어를 따고 → ArtiSynth 입력을 만들고 → (선택) 3D
retargeting/시각화를 하고 → ArtiSynth에서 근육 활성도(inverse)를 구하고 → 정리하는
전체 흐름입니다. 파일 이름의 숫자가 실행 순서입니다.

```
1_extract_contours.py        MRI 마스크 → 혀 표면 컨투어/타깃
2_export_artisynth_inputs.py 컨투어 → ArtiSynth 입력 번들(mri_fit/)
3_kinematic_lift.py          (선택) 대칭 가정 3D lift + GIF
4_retarget_to_artisynth.py   (선택) 실제 ArtiSynth 혀 메시로 retargeting + GIF
5_compare_gif.py             (선택) 원본 MRI vs retarget 비교 GIF
6_static_inverse.py          [ArtiSynth/JPype] 프레임별 정적 inverse → 활성도 CSV
7_summarize_activations.py   활성도 → frame/구간별 표·히트맵
8_forward_from_activations.py [ArtiSynth/JPype] 활성도 → forward FEM mesh
9_compare_forward.py           forward vs retarget round-trip 비교

tongue_contour.py            (모듈) 서브픽셀 컨투어 추출 — 1,2가 import
optical_flow_track.py        (보조) DICOM optical flow → tongue_targets_flow.npy
build_landmark_correspondences.py  (보조) K-point dorsal landmark_map.csv
run_all.py                   1~9 오케스트레이터
artisynth_forward.py         (모듈) JPype in-process forward: muscle_power()
```

> 숫자 접두사가 없는 파일은 **import되는 모듈**(파이썬은 `1_x`를 import 못 함) 또는
> ArtiSynth에서 도는 Jython이라 번호를 안 붙였습니다.

경로는 스크립트 상단 기본값(`E:\...Subject1`, 이 폴더)을 쓰거나 환경변수로 덮어쓸 수
있습니다: `MRI_ROOT`(마스크), `MRI_OUT`(출력=이 폴더), `TONGUE_OBJ`(retarget용 메시).

전체 파이프라인 실행:

```
MRI_SUBJECT=Subject1 python3 run_all.py
```

---

## 공통 좌표계

모든 스텝이 같은 규칙을 씁니다.

| 공간 | x | y (이미지) / z (모델) | y (모델, lateral) |
|------|---|----------------------|-------------------|
| **픽셀** | col (열) | row (행, 아래가 +) | 없음 (2D) |
| **이미지 mm** | `col × 1.164` | `(H-1-row) × 1.164` (위가 +) | z=0 |
| **모델 mm** | anterior→posterior | — | midsagittal y≈0 |
| | | superior = **z up** | 좌우 대칭 |

ArtiSynth `tongue.obj`는 **미터** 단위 → Step 4에서 `×1000`, x에 **+2 mm** offset 적용
(composite jaw+tongue 모델과 맞춤).

스텝 1~4는 **역학(FEM) 없이 운동학(kinematic)** 만 수행합니다. 근육 역학은 Step 6부터
시작합니다.

```
mask (256×256, labels)
    │
    ▼ Step 1: marching squares + airway arc + arc-length 25pts
tongue_targets.npy  (T,25,3) image mm
    │
    ├─► Step 3: symmetric dome lift → tongue_lift_3d.npy (시각화만)
    │
    ├─► Step 2: affine registration (3~K landmarks)
    │         registration.csv, frame_targets_m.csv, contours.csv
    │
    ▼ Step 4: affine map + 13 ctrl delta + RBF skinning
retargeted_tongue.npy  (T,433,3) model mm
```

---

## Step 1 — 컨투어 추출  `1_extract_contours.py`

**목표:** 2D midsagittal RT-MRI 분할 마스크 → 혀 표면 motion target 점열.

```
MRI_SUBJECT=Subject1 python3 1_extract_contours.py
```

### 입력 / 출력

| | |
|---|---|
| **입력** | `MRI_ROOT/mask_*.mat` (라벨 0–6; 혀=4, 기도=5) |
| **출력** | `1_tongue_targets.npy` (T,25,3) dorsal arc |
| | `1_tongue_boundary.npy` (T,80,3) closed outline |
| | `1_landmarks_auto.csv` tip/dorsum/root/floor per frame |
| | `1_qc_trajectories.png`, `1_resampled_markers.png`, `1_qc_boundary_landmarks.png` |
| **의존** | `tongue_contour.py` |

### 알고리즘 A: `precise_contour` — dorsal arc (핵심)

혀 **윗면(dorsum)** 곡선을 tip→root 순으로 25점 추출합니다.

1. **Sub-pixel contour** — `skimage.find_contours(tongue, 0.5)` (marching squares),
   가장 긴 closed loop 선택.
2. **기도-facing arc** — `distance_transform_edt(~airway)`로 기도(5)와 맞닿은 구간만
   남김 (`facing_thresh=2.5 px`). MRI에서 관측 가능한 건 대개 기도 쪽 midsagittal
   실루엣이기 때문.
3. **방향 정렬** — tip = anterior-most = min col → index 0.
4. **Posterior spur clip** (`CLIP_ROOT=1`, 기본 ON) — model에 없는 pharyngeal
   curl-back 제거. dorsum peak에서 root 쪽으로 walk하며 (a) col reversal 또는
   (b) peak 대비 z 하강 (`CLIP_DROP_FRAC≈1.0`) 에서 cut.
5. **Smoothing** — 3-point moving average (끝점 고정).
6. **Arc-length resampling** — 곡선 길이 균등 25점.
7. **이미지 mm 변환** — `x=col×1.164`, `y=(H-1-row)×1.164`, `z=0`.

### 알고리즘 B: `full_boundary_contour` — closed outline

- 같은 marching squares → tip에서 시작, dorsum 방향 orient → arc-length 80점 (loop).

### 알고리즘 C: `anatomical_landmarks`

closed boundary(400점) 위 극값:

| landmark | 정의 |
|----------|------|
| tip | min col (anterior-most) |
| dorsum | min row (superior-most) |
| root | max col (posterior-most) |
| floor | max row (inferior-most) |

> **주의:** landmark는 **full boundary** 기준이고, dorsal arc의 point 0/root end와
> **다를 수 있습니다.** Step 2 registration과 Step 4 retarget이 서로 다른 곡선/점을
> 쓰면 correspondence 오차가 커집니다.

### Step 1의 한계

매 프레임 contour를 **독립 추출** → index `i`가 “같은 조직점”을 보장하지 않음.
`optical_flow_track.py`로 temporal tracking 보완 가능 (`tongue_targets_flow.npy`).

---

## Step 2 — ArtiSynth 입력 만들기  `2_export_artisynth_inputs.py`

**목표:** 2D target → ArtiSynth inverse용 CSV/properties + image↔model registration.

```
MRI_SUBJECT=Subject1 python3 2_export_artisynth_inputs.py
```

K점 registration을 쓰려면 먼저 `build_landmark_correspondences.py` →
`mri_fit/landmark_map.csv` 생성 후 Step 2 재실행.

### 2-1. 전 프레임 contour export

| structure | 추출 | points |
|-----------|------|--------|
| tongue | `precise_contour` (40점, clip 미적용*) | tip→root arc |
| jaw, palate | angle-sorted closed boundary | overlay용 |
| jaw landmark | jaw mask centroid | frame별 |

\*Step 1(25점, clip ON)과 설정이 다름 — `contours.csv`는 40점.

→ `contours.csv`, `landmarks.csv`

### 2-2. Registration — image → model affine

rest frame(기본 frame 1)에서 MRI `(x,y)` → model midsagittal `(x,z)`.

**대응점 (우선순위):**

1. `landmark_map.csv` (사용자 K점)
2. auto: `anatomical_landmarks` tip/dorsum/root ↔ model 고정 좌표
3. fallback: contour arc 극값

**Affine fit (least squares), N≥3:**

```
[ x_img ]       [ modelX ]
[ y_img ]  → A  [ modelZ ]
[   1   ]

A = lstsq( [img_xy | 1], model_xz )
```

model anchor (미터, `tongue.obj` raw): tip `(0.05839, 0.09952)`, dorsum
`(0.09818, 0.11085)`, root `(0.13075, 0.06732)`.

### 2-3. 출력 (`mri_fit/`)

| 파일 | 용도 |
|------|------|
| `registration.csv` | model **mm** (x×1000+2) — **Step 4 retarget** |
| `registration_m.csv` | model **metres** — Step 6 inverse |
| `frame_targets_m.csv` | 프레임별 11 target (metres, y=0) — Step 6 |
| `mri_fit.properties` | JawFemMuscleTongueMriDemo manifest |
| `mri_fit_tongue.properties` | FemTongueMriDemo manifest |

`frame_targets_m.csv` 생성: contour → affine → arc-length 11점 resample, y=0.

Registration은 **rest frame 앵커만** 맞춤. contour 전체 shape은 affine 한 번으로
scale/rotate/translate 수준입니다.

---

## Step 3 — (선택) 운동학 3D lift  `3_kinematic_lift.py`

**목표:** 2D midsagittal curve → **인위적 3D dome** (sanity check). ArtiSynth mesh·FEM
와 무관.

```
MRI_SUBJECT=Subject1 python3 3_kinematic_lift.py
```

### 알고리즘

**입력:** `1_tongue_targets.npy` (T, 25, 2+) — midsagittal curve, tip→root

1. **Width profile** — AP 위치 s∈[0,1]에서 lateral half-width:
   `W(s) = HALF_W × (WIDTH_END + (1-WIDTH_END) × sin(πs)^0.6)` (tip/root 좁음,
   mid-body 넓음).
2. **Lateral sampling** — 각 midsagittal 점에서 z_lateral ∈ [-W, +W]를 NZ=15개.
3. **Coronal dome** — `drop = EDGE_DROP × (1 - sqrt(1 - (z_lat/W)²))`, y_3d = y_crest - drop.

**출력:** `3_tongue_lift_3d.npy` (T,25,15,3), `3_lift_frames3d.png`, `3_lift_motion.gif`

lateral profile은 **실측 아님** (대칭 dome 가정). Step 4 retarget과 직접 연결되지 않음.

---

## Step 4 — ArtiSynth 혀 메시 retargeting  `4_retarget_to_artisynth.py`

**목표:** MRI motion → **실제 `tongue.obj` mesh** (433 verts) kinematic 변형.
Gaussian RBF skinning.

```
MRI_SUBJECT=Subject1 python3 4_retarget_to_artisynth.py
# optical-flow target 사용 시:
TARGETS_NPY=tongue_targets_flow.npy python3 4_retarget_to_artisynth.py
```

### 입력 / 출력

| | |
|---|---|
| **입력** | `tongue_targets.npy` (또는 flow 버전), `mri_fit/registration.csv`, `TONGUE_OBJ` |
| **출력** | `4_retargeted_tongue.npy` (T,433,3) mm |
| | `4_retarget_midsag.png`, `4_retarget_frames3d.png`, `4_retarget_motion.gif` |
| | `4_retargeted_objs/frame_*.obj` |

### 4-1. Model mesh 로드

`tongue.obj` vertices ×1000 (mm), x += 2 mm.

### 4-2. Model dorsal control curve (13점)

midsagittal verts (|y|<3 mm)에서 x를 13 bin으로 나누고 각 bin의 z_max → dorsal
envelope, 3-point smoothing.

### 4-3. MRI → model frame

`registration.csv` anchors로 affine fit → 각 frame 25 dorsal target을 model (x,z) mm로
변환.

### 4-4. Control displacement

```python
mri[k] = resample(mapped_25pts, NCTRL=13)   # arc-length 13점 (현재 구현)
delta[k] = mri[k] - mri[REST]              # rest 대비 변위; REST에서 delta=0
```

### 4-5. Smoothing

- **Spatial:** `uniform_filter1d` window=3 (curve 위)
- **Temporal:** Savitzky-Golay window=9, poly=2 (~1.8 s @ 5 fps)
- 다시 `delta -= delta[REST]`로 rest anchor

### 4-6. Gaussian RBF skinning

각 frame k:

```
RBFInterpolator(
    centers = dorsal (13 rest points, x-z),
    values  = delta[k],
    kernel  = Gaussian, epsilon = 1/RBF_LEN  (RBF_LEN=18 mm)
)
→ 모든 vertex (x,z)에 displacement 전파; y는 mesh+RBF가 간접 채움
```

### Step 4 한 줄 요약

> rest affine으로 MRI dorsal motion을 model (x,z)로 옮기고, model dorsal 13점 delta를
> Gaussian RBF로 433 vertex 전체에 skinning.

### 알고리즘상 약한 고리 (correspondence)

| # | 문제 | 내용 |
|---|------|------|
| 1 | **Temporal 2D** | Step 1 per-frame resample → index i ≠ same tissue → flow 추적 |
| 2 | **Spatial 2D↔3D** | registration=anatomical landmarks, retarget=dorsal arc; curve 불일치 |
| 3 | **Parameterization** | MRI 13점=arc-length, model 13점=x-bin → index pairing mismatch |

개선 방향: `build_landmark_correspondences.py` + `resample_by_x`, optical flow
(`--mode direct`), AP extent crop. 자세한 배경은 `SKILL.md`, `README_registration_transfer.md`.

---

## Step 5 — (선택) 비교 GIF  `5_compare_gif.py`

원본 MRI(분할+추적 마커)와 retarget된 혀를 좌우로 붙인 단일 GIF.

- 입력: `tongue_targets.npy`, `retargeted_tongue.npy`, 마스크, `TONGUE_OBJ`
- 출력: `compare_mri_vs_retarget.gif`

```
python 5_compare_gif.py
```

## Step 5 — (선택) 비교 GIF  `5_compare_gif.py`

원본 MRI(분할+추적 마커)와 retarget된 혀를 좌우로 붙인 단일 GIF.

- 입력: `tongue_targets.npy`, `retargeted_tongue.npy`, 마스크, `TONGUE_OBJ`
- 출력: `compare_mri_vs_retarget.gif`

```
python 5_compare_gif.py
```

## Step 6 — 근육 활성도 inverse (ArtiSynth)

활성도는 ArtiSynth FEM 솔버의 출력이라 **ArtiSynth에서** 구합니다. 두 가지 방법:

### 6a. 동적 tracking (관성 포함, 연속)
ArtiSynth에서 모델을 manifest와 함께 로드:
```
bin\artisynth.bat -model artisynth.models.tongue3d.FemTongueMriDemo [ <...>\mri_fit\mri_fit_tongue.properties ]
```
(턱+혀는 `artisynth.models.jawTongue.JawFemMuscleTongueMriDemo` + `mri_fit.properties`)
→ `mriTracking` 패널 확인 → 정지시각 21초 → Play → **File → Save output probe data**
→ `subject1_computed_excitations.txt`(활성도).

### 6b. 프레임별 독립 정적 inverse

**Docker (GUI 없음, 권장):**

```bash
docker compose build tongue-artisynth    # 최초 1회
docker compose run --rm inverse-test     # 1프레임 테스트
docker compose run --rm inverse          # 전체 → activations_static_per_frame.csv
```

**ArtiSynth GUI + Jython** (`6_static_inverse.py`):

1. FemTongueMriDemo + manifest 로드 (`mriTracking` 패널 확인)
2. **Scripts → Run Script… → `6_static_inverse.py`**

**Python API (호스트/WSL, JDK+ArtiSynth 필요):**

```bash
pip install JPype1
ARTISYNTH_HOME=/opt/artisynth/artisynth_core python3 artisynth_static_inverse.py
```

자세한 Docker 설정: `README.docker.md`

## Step 7 — 활성도 정리  `7_summarize_activations.py`

ArtiSynth가 쓴 활성도 텍스트를 frame/구간별 표·히트맵으로 변환.

```
python 7_summarize_activations.py mri_fit/subject1_computed_excitations.txt --fps 5 --segments 7
# 또는 6b 결과(이미 표 형식)인 mri_fit/activations_static_per_frame.csv 사용
```
- 출력: `activations_per_frame.csv`, `activations_per_interval.csv`,
  `activations_heatmap.png`, `activations_peaks.png`

---

## (선택) Forward 모델 — `muscle_power(activations) → 3D mesh`

근육값을 주면 ArtiSynth 솔버가 혀를 변형해 메시를 돌려줍니다(Step 6의 반대 방향).

- **소켓**: ArtiSynth에서 `forward_server.py` 실행(Scripts → Run Script) → 파이썬에서
  `from tongue_forward import muscle_power`.
- **in-process(JPype)**: `from artisynth_forward import init, muscle_power`
  (Java+ArtiSynth가 같은 머신에 있어야 함).

자세한 건 `README.docker.md`(도커 all-in-one)와 각 파일 상단 주석 참고.

---

## 의존 관계 요약

```
1 ──> tongue_targets.npy ──> 2 ──> mri_fit/ (registration, manifests, frame_targets)
                         ├─> 3 (lift, sanity)
                         └─> 4 (retarget) ──> 5 (compare MRI vs retarget)
2 ──> 6 (static inverse) ──> 7 (activations summary)
                         └─> 8 (forward) ──> 9 (round-trip vs retarget)
optical_flow_track.py ──> tongue_targets_flow.npy ──> 4/5 (optional)
build_landmark_correspondences.py ──> landmark_map.csv ──> 2 (optional)
```

## 관련 문서
- `SKILL.md` — 파이프라인 규칙, gotcha, correspondence 한계
- `README_registration_transfer.md` — 설계/배경, ArtiSynth GUI 실행·문제해결 상세
- `README.docker.md` — 도커 환경(파이썬 파이프라인 + ArtiSynth forward all-in-one)
