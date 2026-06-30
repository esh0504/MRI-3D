# SKILL.md: 2D MRI 기반 3D 생체역학 모델 역산 및 시계열 최적화 파이프라인

## 📌 Project Overview
본 프로젝트는 3D Ground Truth(GT)가 부재한 2D MRI 비디오 시퀀스 환경에서, 3D 생체역학 혀 모델(Artisynth)의 11차원 제어점(근육 활성값)을 역산하는 Analysis-by-Synthesis 최적화 파이프라인 구축을 목표로 합니다. 블랙박스 시뮬레이터 환경에서 발생하는 국소 최적해(Local Minima) 및 차원의 저주를 DFO(Derivative-Free Optimization)와 시계열 정규화 기법으로 극복합니다.

---

## 🛠️ Core Tech Stack & Dependencies
* **Simulation Engine:** Artisynth (Java)
* **Optimization Algorithm:** CMA-ES (`cma` Python library)
* **Sampling Strategy:** LHS (Latin Hypercube Sampling) / Sobol Sequence
* **Deep Learning Framework (Phase 4):** PyTorch
* **Communication Interface:** Socket Programming / CLI I/O (Python ↔ Java Bridge)

---

## 🚀 Research Roadmap & Milestones

### Phase 1: 기반 인프라 구축 및 초기값 사전(Dictionary) 생성
최적화 알고리즘의 안정적인 수렴을 위한 통신 브릿지 구축 및 탐색 공간(Search Space) 축소 단계.

* [x] **Python-Artisynth 통신 브릿지 구축:** 11D 벡터 전송 및 가상 2D Mask 반환 I/O 파이프라인 완성.
* [ ] **다차원 공간 샘플링:** 11차원 제어 공간을 라틴 하이퍼큐브 샘플링(LHS)으로 균일하게 분할.
* [ ] **Coarse DB 렌더링:** 병렬 처리를 통해 `[11D 활성값] ↔ [2D 가상 실루엣]` 매핑 데이터셋 구축 (최소 5만 개 이상).
* [ ] **유사도 매칭 모듈 구현:** 2D 실루엣 간의 형태 오차(Chamfer Distance, IoU) 계산 함수 작성.

### Phase 2: 단일 프레임 최적화 (DFO 설계 및 검증)
시간 축을 배제한 정지된 단일 프레임 환경에서 형태학적 손실 함수를 통한 11D 역산 모듈 완성.

* [ ] **DFO 코어 세팅:** `cma` 라이브러리를 활용한 전역 최적화 루프 구성.
* [ ] **목적 함수(Loss) 설계:** * `E_shape`: MRI와 가상 Mask 간의 형태 매칭 오차.
    * `L2 Regularization`: 생체 역학의 근육 중복성(Null Space) 문제를 해결하기 위한 최소 노력의 원리 적용.
* [ ] **하이브리드 초기화 (Hybrid Initialization):** 맨땅 탐색 방지를 위해 Phase 1의 DB에서 Top-K 후보를 추출하여 CMA-ES의 초기 군집(Initial Guess)으로 주입.
* [ ] **단일 프레임 수렴 검증:** 목표 실루엣에 대한 Global Minimum 도달 여부 및 재현성 테스트.

### Phase 3: 시계열 동역학(Temporal Dynamics) 확장
혀의 점탄성, 관성 및 물리적 연속성을 반영하여 지터링(Jittering) 없는 부드러운 궤적(Trajectory) 추출.

* [ ] **순차적 초기화 적용:** $t$ 프레임 최적화 시, 이전 프레임 $t-1$의 도출값을 초기 평균점으로 설정하여 탐색 속도 극대화.
* [ ] **시계열 정규화 (Temporal Smoothness):** 목적 함수에 속도/가속도 페널티를 추가하여 물리적으로 타당한 움직임 강제.
* [ ] **Sliding Window 최적화 (Local BA):** 3~5개 프레임 단위의 미니 전역 최적화를 주기적으로 수행하여 오차 누적(Drift) 방지.

### Phase 4: 딥러닝 기반 대리 모델(Surrogate) 고도화
실시간 역산(Real-time Inference) 달성 및 알고리즘 의존도 탈피를 위한 AI 모델 학습.

* [ ] **Pseudo-GT 시퀀스 확보:** Phase 3의 최적화 알고리즘을 장기 렌더링하여 다량의 2D MRI ↔ 11D 활성값 시퀀스 데이터셋 확보.
* [ ] **Temporal Model 설계:** TCN(Temporal Convolutional Network) 또는 Video Transformer 아키텍처 설계.
* [ ] **End-to-End 회귀 학습:** 2D MRI 시퀀스를 입력받아 시뮬레이터 없이 11D 활성값 시퀀스를 직접 추론하는 네트워크 구축 및 성능 평가.