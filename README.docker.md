# Docker — ArtiSynth 올인원 (설치 + forward + inverse)

컨테이너 **안에** ArtiSynth를 clone·컴파일해 넣습니다.  
호스트에 Java/ArtiSynth를 따로 설치하거나 마운트할 필요 **없음**.

---

## 1. 빌드 (최초 1회, 수 분)

```bash
cd /path/to/Tongue_Inverse   # 이 프로젝트 루트
docker compose build tongue-artisynth
```

빌드 중 자동으로:

1. `github.com/artisynth/artisynth_core` + `artisynth_models` clone
2. models 소스를 core `src/`에 병합 (`HexTongueDemo`, `JawHyoidFemMuscleTongue` 등)
3. Linux 네이티브 솔버 (`lib/Linux64/*.so`) 다운로드
4. `artisynth/src/` 커스텀 MRI/inverse Java 컴파일
4. Python + JPype 설치

---

## 2. 정적 Muscle Inverse (Step 6b)

**사전:** `mri_fit/frame_targets_m.csv` 가 있어야 함 (Step 2 실행 또는 저장소에 포함).

```bash
# 전체 프레임 (105프레임 × 0.4s — 시간 오래 걸림)
docker compose run --rm inverse

# 1프레임 연결 테스트
docker compose run --rm inverse-test
```

출력: `mri_fit/activations_static_per_frame.csv`

환경변수 튜닝:

```bash
docker compose run --rm -e MAX_FRAMES=3 -e SETTLE_T=0.4 inverse
```

| 변수 | 기본 | 의미 |
|------|------|------|
| `SETTLE_T` | 0.4 | 프레임당 settle 시간(초) |
| `MAX_FRAMES` | 0 | 0=전체, N=N프레임만 |
| `JVM_XMX` | 4g | JVM 힙 |
| `MRI_MANIFEST` | `/work/mri_fit/mri_fit_tongue.properties` | manifest |
| `TARGETS_CSV` | `/work/mri_fit/frame_targets_m.csv` | 타깃 입력 |
| `OUT_CSV` | `/work/mri_fit/activations_static_per_frame.csv` | 출력 |

Step 7 정리 (호스트 또는 컨테이너):

```bash
docker compose run --rm tongue-artisynth \
  python3 7_summarize_activations.py mri_fit/activations_static_per_frame.csv --fps 5 --segments 7
```

---

## 3. Forward — `muscle_power(a) -> mesh`

```bash
docker compose run --rm tongue-artisynth python3 - <<'PY'
from artisynth_forward import init, muscle_power, save_obj, shutdown
names = init()
print("muscles:", names)
v, f = muscle_power([0.3] + [0.0] * (len(names) - 1))
save_obj(v, f, "/work/pose_test.obj")
print("wrote /work/pose_test.obj", v.shape)
shutdown()
PY
```

---

## 4. MRI 마스크에서 처음부터 (Step 1–2)

마스크 폴더가 있으면 `.env` 설정:

```bash
cp docker.env.example .env
# .env 에 MRI_MASKS=/path/to/Subject1
```

`docker-compose.yml` 의 masks 볼륨 주석 해제:

```yaml
- ${MRI_MASKS}:/data/masks:ro
```

```bash
docker compose run --rm tongue-artisynth python3 2_export_artisynth_inputs.py
docker compose run --rm inverse
```

---

## 5. 셸 진입

```bash
docker compose run --rm tongue-artisynth
# 컨테이너 안에서 python3 artisynth_static_inverse.py 등
```

---

## 6. 문제 해결

| 증상 | 해결 |
|------|------|
| `manifest not found` | Step 2 실행 또는 `mri_fit/` 확인 |
| `TrackingController not found` | 이미지 재빌드 (`FemTongueMriDemo` 미컴파일) |
| `UnsatisfiedLinkError` | `docker compose build` 재실행 (네이티브 lib) |
| 매우 느림 | `inverse-test`로 1프레임 먼저, `MAX_FRAMES`로 부분 실행 |
| 빌드 실패 `cannot find symbol HexTongueDemo` | `artisynth_models` 미병합 — 이미지 재빌드 (`Dockerfile.artisynth` 최신 확인) |
| 빌드 실패 (git/network) | 인터넷 확인, `ARTISYNTH_GIT` / `ARTISYNTH_MODELS_GIT` build-arg |

---

## 7. 대안: 호스트 ArtiSynth + 소켓 (`tongue` 서비스)

ArtiSynth GUI/Windows를 그대로 쓰고 Python만 Docker:

- 호스트: `forward_server.py` 실행
- `docker compose run --rm tongue` + `tongue_forward.py`

inverse는 **`inverse` 서비스**(컨테이너 내 ArtiSynth) 사용을 권장.
