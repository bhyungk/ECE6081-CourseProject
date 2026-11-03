# WESAD 스트레스 조건 분류 시작 코드

- 제공 범위: 데이터 로딩, 특징 계산, 참가자 분할, Logistic Regression 평가, 결과 저장
- 필수 내용: Advanced Model 개발, 기준 모델과의 비교, RQ1~RQ4 실험·분석

## 설치

- 검증 환경: Python 3.12, macOS arm64 ([고정 버전 목록](requirements-lock.txt))
- 데이터: [WESAD.zip 다운로드](https://drive.google.com/file/d/14oqZSredIiGFY_tcMJ4C_TXWln7qguzl/view?usp=sharing) (약 2.25 GB), 압축 해제
- 폴더 구조: `$WESAD_ROOT/S2/S2.pkl` 등 15명(S2 ~ S11, S13 ~ S17)
- `WESAD_ROOT`: 압축 해제 후 `S2`, `S3` 등 참가자 폴더가 바로 들어 있는 경로
- 제공 결과 확인·저장된 특징 재평가: 원본 데이터 불필요
- 원신호 특징 추출·RQ2의 BVP 잡음 실험: 원본 데이터 필요

### 저장소 받기

```bash
git clone https://github.com/bhyungk/ECE6081-CourseProject.git
cd ECE6081-CourseProject
```

### macOS arm64 (검증 환경)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

### Linux / WSL

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

- Linux·Windows: 운영체제에 맞는 패키지 버전 설치, 실행 환경의 버전 기록
- Windows: 가상 환경 활성화 없이 `.\.venv\Scripts\python.exe` 사용

## 실행

- 실행 위치: 저장소 최상위 폴더
- `WESAD_ROOT`: 압축을 푼 본인의 WESAD 폴더 경로
- `run`: 원본 특징 추출 → 전체 평가 → 결과 저장·검증
- `evaluate`: 저장된 특징으로 기준 모델 재평가
- `verify`: 저장된 예측·지표·분할의 일관성 재확인

### 저장된 특징으로 재평가

```bash
python -m wesad_starter evaluate --features results/baseline/features.csv --output-dir results/recheck
python -m wesad_starter verify --output-dir results/recheck
```

- Windows: 위 명령의 `python`을 `.\.venv\Scripts\python.exe`로 변경
- 저장된 특징·기준 결과: `results/baseline/` / 재평가 결과: `results/recheck/`

### 원본 데이터부터 실행

macOS / Linux / WSL:

```bash
export WESAD_ROOT="$HOME/ResearchDatasets/WESAD"
python -m wesad_starter run --data-root "$WESAD_ROOT" --output-dir results/my_baseline
python -m wesad_starter verify --output-dir results/my_baseline
```

Windows PowerShell:

```powershell
$env:WESAD_ROOT = "C:\Users\사용자이름\ResearchDatasets\WESAD"
.\.venv\Scripts\python.exe -m wesad_starter run --data-root "$env:WESAD_ROOT" --output-dir results/my_baseline
.\.venv\Scripts\python.exe -m wesad_starter verify --output-dir results/my_baseline
```

- 출력: `results/my_baseline/` / 경로는 본인 환경에 맞게 변경

### 노트북

- JupyterLab 설치: `python -m pip install jupyterlab`
- 실행: `python -m jupyter lab notebooks/01_baseline_walkthrough.ipynb`
- Windows: 위 명령의 `python`을 `.\.venv\Scripts\python.exe`로 변경
- 기본 입력: `results/baseline/` / 다른 결과 사용 시 `WESAD_RESULTS` 설정
- 원신호 예시: `WESAD_ROOT`와 `WESAD_DEMO_RAW=1` 설정

## 공통 실험 설정

- 입력: 손목 EDA 4 Hz, BVP 64 Hz / 원래 조건 라벨 700 Hz
- 구간: 기록 시작점 기준 60초 비중첩, 원래 조건이 일정한 1·2·3 구간만 사용
- 정답: Stress(2)=1, Baseline(1)·Amusement(3)=0 / 원래 조건 검사 후 이진 변환
- 특징: 신호별 평균·표준편차·최솟값·최댓값·중앙값·IQR·기울기 7개
- 평가: 15명 LOSO, 네 조건에서 동일 구간 사용
- 전처리: 학습 참가자의 중앙값 대체 → StandardScaler / 전체 결측 열은 0 대체·유지
- 기준 모델: Logistic Regression, `C=1`, `class_weight="balanced"`, `max_iter=2000`, 임곗값 0.5
- 비교 조건: EDA, BVP, 특징 연결 융합, EDA·BVP 확률의 0.5:0.5 평균 융합
- 주 지표: 참가자별 Macro-F1의 평균·표본 표준편차(`ddof=1`)
- 누수 방지: 메타데이터 입력 제외, 전처리·모델 선택에 평가 참가자 정보 사용 금지

## 결과

- 저장 위치: `results/baseline/`
- `window_manifest.csv`, `features.csv`: 구간 포함·제외 기록과 특징
- `predictions.csv`: 구간별 정답·확률·예측
- `participant_metrics.csv`, `summary.csv`: 참가자별 점수와 전체 요약
- `fusion_deltas.csv`, `condition_errors.csv`, `error_overlap.csv`: 융합 차이·조건별 오류·오류 중복
- `splits.json`, `preprocessing.csv`, `*_metadata.json`: 참가자 분할·전처리·실행 설정
- 기준 개수: 513구간(Baseline 282 / Stress 152 / Amusement 79), 예측 2,052행, 참가자 점수 60행

## 프로젝트 요구사항

- Advanced Model: 기준 모델과 차별화되는 개선 요소 1개 이상 구현, 동일한 네 조건 비교
- 모델 선택: 특징 추출·모델 구조·융합 방식 자유 설계, 머신러닝·딥러닝 모두 가능
- 특징 융합: 신호별 특징 연결 또는 학습된 표현 결합 가능
- 제공 코드: 기준 모델과 RQ1의 기초 분석 / Advanced Model과 RQ2~RQ4는 팀별 직접 구현
- RQ1: 참가자별 융합 이득·악화와 단일 신호 오류 중복 분석
- RQ2: BVP 원신호에 잡음 두 수준 적용 후 특징·표현 재계산, 손상 전후 성능 비교
- RQ2 기록: 잡음 종류·강도·적용 시점·난수 설정·재학습 여부
- RQ3: 동일 가중치와 외부 학습 참가자의 내부 검증으로 선택한 가중치 비교
- RQ4: EDA 2모델·BVP 2모델 앙상블과 EDA+BVP 예측 융합 비교
- RQ4 통제: 동일 평가 구간·모델 수, 가능한 한 같은 학습 예산 / 차이 기록
- RQ4 예시: 외부 학습 참가자의 복원 추출 또는 확률적 초기화로 구성원 다양화
- RQ4 유의: Logistic Regression은 seed만 바꿔도 예측이 같을 수 있음 / 구성원별 예측 확률·학습 조건 기록
- RQ4 해석: 외부 평가 결과로 구성원을 선택하지 않음 / 상호 보완성·인과관계 주장은 확인한 근거로 제한
- 내부 검증: 외부 학습 참가자 안에서 참가자 단위 K-fold 등 사용 / fold 수와 탐색 범위 조절 가능
- 누수 방지: 각 내부 분할의 전처리는 내부 학습 참가자로만 추정 / 외부 평가 참가자로 설정·가중치 선택 금지
- 최종 평가: 설정 선택 후 외부 학습 참가자 전체로 재학습 → 남겨 둔 참가자로 평가
- 최종 제출: 본문 8쪽 이내 보고서(참고문헌·상세 설정 부록 별도), 실행 코드, 결과 CSV

## 참고

- [설명 노트북](notebooks/01_baseline_walkthrough.ipynb): 저장 결과 확인·평가 재실행
- [특징 명세](docs/feature_spec.md): 계산식·결측 처리
- [검증 기록](docs/validation.md): 실제 데이터 기준 결과·환경 재현
- [WESAD 공식 페이지](https://kristofvl.github.io/usi/data_wesad.html) · [원 논문](https://doi.org/10.1145/3242969.3242985): 데이터 설명과 출처
- 테스트 실행: `python -m pytest -q`
