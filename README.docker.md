# Docker 환경 — ArtiSynth forward 모델을 컨테이너에서

`muscle_power(activations) -> 3D mesh` 를 **한 컨테이너 안**에서 돌리는 환경입니다.
실제 ArtiSynth FEM 솔버(리눅스 네이티브 Pardiso)를 써서, 근육 활성값을 주면 혀가
변형된 표면 메시를 돌려줍니다.

핵심 아이디어: ArtiSynth를 새로 빌드하지 않습니다. 당신의 **이미 컴파일된 ArtiSynth
트리를 마운트**하고(자바 bytecode라 리눅스에서 그대로 실행), 플랫폼 전용인 **리눅스
네이티브 솔버(.so)만 컨테이너 첫 실행 때 자동으로 받습니다.**

---

## 1. 준비물

- Docker Desktop (Windows/Mac) 또는 Docker (Linux)
- 컴파일된 ArtiSynth 트리: `C:\Users\d11\artisynth` (안에 `artisynth_core/classes`, `lib`, `bin`)
- (선택) MRI 마스크 폴더: `E:\Datasets\XAI\data\GT_Segmentations\Subject1`
- 인터넷 (첫 실행 때 리눅스 네이티브 솔버 다운로드)

구성 파일: `Dockerfile.artisynth`, `docker-entrypoint.sh`, `docker-compose.yml`,
`artisynth_forward.py`, `tongue_forward.py`.

---

## 2. 빌드 & 실행

```
cd C:\Users\d11\Project\Tongue_Inverse
docker compose build tongue-artisynth        # 가벼움: JDK + python + jpype 만
docker compose run --rm tongue-artisynth      # 컨테이너 진입
```

컨테이너가 처음 뜰 때 **엔트리포인트**가 자동으로 리눅스 네이티브 솔버를 받습니다
(`lib/Linux64/*.so`). 콘솔에 다음이 보이면 준비 완료:

```
[setup] Linux native libs missing -> fetching from artisynth.org (one-time)...
[setup] native libs ready in /opt/artisynth/artisynth_core/lib/Linux64
```

(이미 받았으면 `[setup] Linux native libs present` 로 건너뜀.)

---

## 3. 사용 — `muscle_power(a) -> mesh`

컨테이너 안에서:

```python
python3 - <<'PY'
from artisynth_forward import init, muscle_names, muscle_power, save_obj
names = init()                       # JVM 시작 + 모델 빌드 + 솔버 로드
print(names)                         # ['GGP','GGM','GGA','STY','GH','MH','HG','VERT','TRANS','IL','SL']
verts, faces = muscle_power([0.3] + [0.0]*10)   # GGP 0.3, 나머지 0
print(verts.shape)                   # (Nverts, 3) 변형된 혀 표면 (모델 미터)
save_obj(verts, faces, "/work/pose.obj")
PY
```

- 입력은 리스트(근육 순서대로) 또는 dict: `muscle_power({"GGP":0.3, "HG":0.2})`.
- 호출마다: rest로 reset → 활성도를 0→목표로 **점진 적용(ramp)** → 평형까지 forward
  solve → 변형 메시 반환. (램프/작은 스텝으로 요소 반전 방지.)

움직임 확인:

```python
python3 - <<'PY'
from artisynth_forward import init, muscle_power
import numpy as np
init()
r,_ = muscle_power([0]*11)
v,_ = muscle_power([0.3]+[0]*10)
print("max disp mm:", float(np.abs(v-r).max()))   # 수 mm면 근육이 혀를 움직인 것
PY
```

---

## 4. 튜닝 (환경변수)

| 변수 | 기본 | 의미 |
|---|---|---|
| `SETTLE_T` | 0.4 | 호출당 평형까지 시뮬 시간(초). 부족하면 ↑ |
| `NRAMP` | 20 | 활성도 램프 단계 수. 요소 반전 나면 ↑ |
| `MAXSTEP` | 0.005 | 솔버 최대 스텝(초). 불안정하면 ↓ |
| `INCOMP` | OFF | 비압축성 방식. **OFF가 안정**(AUTO는 부하 시 요소 반전). 필요시 AUTO/ELEMENT/NODAL |
| `JVM_XMX` | 4g | JVM 최대 힙 |
| `TONGUE_MODEL` | `...tongue3d.HexTongueDemo` | 사용할 FEM 근육 혀 모델 |
| `ARTISYNTH_HOME` | `/opt/artisynth/artisynth_core` | 마운트된 트리 |

예: `docker compose run --rm -e NRAMP=40 -e MAXSTEP=0.002 tongue-artisynth`

---

## 5. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `ClassNotFoundException: ...HexTongueDemo` | ArtiSynth 트리 마운트 안 됨 → compose의 `/opt/artisynth` 마운트 확인 |
| `syntax error near 'else'`, `'in/...` | 셸 스크립트가 Windows CRLF. 엔트리포인트는 java를 직접 호출하므로 무관 |
| `UnsatisfiedLinkError` / Pardiso 로드 실패 | 네이티브 미페치. 마운트가 **read-write**인지, 인터넷 되는지 확인 후 재실행 |
| `NumericalException: Inverted elements` | 활성도 급가 → `NRAMP`↑, `MAXSTEP`↓, 또는 활성도 낮추기 |
| 첫 실행이 네이티브 다운로드에서 멈춤 | 정상(다운로드 중). 끝나면 진행 |

네이티브 라이브러리는 당신 트리의 `lib/Linux64/` 에만 추가되며 Windows 동작(`lib/Windows64/`)에는 영향 없습니다.

---

## 6. 대안: 경량 컨테이너 + 호스트 ArtiSynth (소켓)

ArtiSynth를 컨테이너에 넣고 싶지 않다면, 호스트(Windows)에서 ArtiSynth를 그대로 쓰고
파이썬만 slim 컨테이너에서 돌릴 수 있습니다:

- 호스트: ArtiSynth에서 `forward_server.py` 실행 (`Listening on port 5005`).
- 컨테이너(`tongue` 서비스, `Dockerfile`): `tongue_forward.py` 가
  `host.docker.internal:5005` 로 접속.

```
docker compose run --rm tongue
python3 -c "from tongue_forward import muscle_power; print(muscle_power([0.3]+[0]*10)[0].shape)"
```

두 방식 모두 같은 ArtiSynth 솔버라 결과는 동일합니다. (자세한 분석 스크립트·inverse는
`README.md` 참고.)
