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
6_static_inverse.py          [ArtiSynth/Jython] 프레임별 정적 inverse → 활성도 CSV
7_summarize_activations.py   활성도 → frame/구간별 표·히트맵

tongue_contour.py            (모듈) 서브픽셀 컨투어 추출 — 1,2가 import
forward_server.py            (선택, ArtiSynth/Jython) 근육→형상 forward 소켓 서버
tongue_forward.py            (선택, 모듈) forward 서버 클라이언트: muscle_power()
artisynth_forward.py         (선택, 모듈) JPype in-process forward: muscle_power()
```

> 숫자 접두사가 없는 파일은 **import되는 모듈**(파이썬은 `1_x`를 import 못 함) 또는
> ArtiSynth에서 도는 Jython이라 번호를 안 붙였습니다.

경로는 스크립트 상단 기본값(`E:\...Subject1`, 이 폴더)을 쓰거나 환경변수로 덮어쓸 수
있습니다: `MRI_ROOT`(마스크), `MRI_OUT`(출력=이 폴더), `TONGUE_OBJ`(retarget용 메시).

---

## Step 1 — 컨투어 추출  `1_extract_contours.py`

MRI 마스크(`mask_*.mat`, 라벨 0–6)에서 혀(4)↔기도(5) 접면 표면을 **서브픽셀**로 따서
tip→root로 정렬·리샘플합니다.

- 입력: `MRI_ROOT`의 `mask_*.mat`
- 출력: `tongue_targets.npy` (T,25,3), `tongue_targets.txt`(probe), `qc_trajectories.png`,
  `resampled_markers.png`
- 의존: `tongue_contour.py`

```
python 1_extract_contours.py
```

## Step 2 — ArtiSynth 입력 만들기  `2_export_artisynth_inputs.py`

컨투어를 ArtiSynth inverse가 먹는 포맷으로 내보냅니다(+ 이미지→모델 정합 앵커).

- 출력(`mri_fit/`):
  - `contours.csv`, `landmarks.csv`
  - `registration.csv`(mm) + `mri_fit.properties` → **JawFemMuscleTongueMriDemo**(턱+혀)
  - `registration_m.csv`(m) + `mri_fit_tongue.properties` → **FemTongueMriDemo**(혀, 미터)
  - `frame_targets_m.csv` → Step 6(정적 inverse)용 프레임별 11개 타깃(모델 미터)

```
python 2_export_artisynth_inputs.py
```

## Step 3 — (선택) 운동학 3D lift  `3_kinematic_lift.py`

대칭 가정으로 정중시상 곡선을 3D 돔 표면으로 lift(역학 없음, sanity check).

- 입력: `tongue_targets.npy`
- 출력: `tongue_lift_3d.npy`, `lift_frames3d.png`, `lift_motion.gif`

```
python 3_kinematic_lift.py
```

## Step 4 — (선택) ArtiSynth 혀 메시로 retargeting  `4_retarget_to_artisynth.py`

실제 ArtiSynth 혀 메시(`tongue.obj`)를 MRI 운동에 맞춰 대칭 RBF skinning으로 변형
(운동학, 시간 스무딩 포함).

- 입력: `tongue_targets.npy`, `mri_fit/registration.csv`, `TONGUE_OBJ`
- 출력: `retargeted_tongue.npy`, `retarget_midsag.png`, `retarget_frames3d.png`,
  `retarget_motion.gif`, `retargeted_objs/frame_*.obj`

```
python 4_retarget_to_artisynth.py
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

### 6b. 프레임별 독립 정적 inverse  `6_static_inverse.py` (Jython)
각 프레임을 독립적으로 평형까지 풀어 활성도를 기록.
1. 위 6a처럼 모델을 로드(`mriTracking` 패널이 떠 있어야 함).
2. ArtiSynth **Scripts → Run Script… → `6_static_inverse.py`**.
3. 출력: `mri_fit/activations_static_per_frame.csv` (frame,time,근육별 활성도).

> ArtiSynth 실행/빌드/문제해결 상세는 `README_registration_transfer.md`,
> 도커로 ArtiSynth forward까지 돌리는 법은 `README.docker.md` 참고.

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
                         └─> 3 (lift)
                         └─> 4 (retarget) ──> 5 (compare)
2 ──> ArtiSynth inverse (6a 동적 / 6b 정적) ──> activations*.txt/csv ──> 7 (정리)
```

## 관련 문서
- `README_registration_transfer.md` — 설계/배경, ArtiSynth GUI 실행·문제해결 상세
- `README.docker.md` — 도커 환경(파이썬 파이프라인 + ArtiSynth forward all-in-one)
