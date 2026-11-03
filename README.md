# WESAD 스트레스 조건 분류 시작 코드

WESAD 손목 EDA와 BVP를 이용해 새로운 참가자의 스트레스 조건을 분류하는 대학원 프로젝트입니다. 제공된 Logistic Regression 기준 모델을 출발점으로 **각 팀의 Advanced Model을 설계·구현하고, 기준 모델과 비교하며 RQ1~RQ4를 모두 수행**합니다.

## 데이터 준비

1. [WESAD 원본 데이터 다운로드](https://drive.google.com/file/d/14oqZSredIiGFY_tcMJ4C_TXWln7qguzl/view?usp=sharing)에서 `WESAD.zip`을 내려받고 압축을 풉니다. 압축 파일은 약 2.25 GB이며 압축 해제에 추가 공간이 필요합니다.
2. 압축을 푼 뒤 아래와 같이 참가자 폴더 `S2`, `S3` 등이 **바로 들어 있는 폴더**를 찾습니다. 이 폴더가 `WESAD_ROOT`입니다. 압축 해제 과정에서 `WESAD/WESAD/`처럼 폴더가 중첩되었다면 안쪽 폴더를 지정합니다.

```text
WESAD_ROOT/
├── S2/S2.pkl
├── S3/S3.pkl
├── ...
└── S17/S17.pkl
```

전체 실험에는 S2~S11과 S13~S17, 총 15명의 동기화 `.pkl` 파일이 필요합니다. WESAD에 대한 자세한 설명과 데이터 이용 안내는 [공식 페이지](https://kristofvl.github.io/usi/data_wesad.html)와 [원 논문](https://doi.org/10.1145/3242969.3242985)을 확인하세요. 원본 데이터는 GitHub에 다시 올리지 마세요.

저장소의 `results/baseline/`에는 실제 WESAD로 계산한 특징과 기준 결과가 포함되어 있습니다. **기존 결과 확인과 저장된 특징을 이용한 기준 평가 재실행에는 원본이 필요하지 않습니다.** 원신호부터 특징을 다시 추출하거나 RQ2에서 BVP 원신호에 잡음을 적용하려면 원본 데이터가 필요합니다.

## 설치

Python 3.12를 권장합니다. 아래 명령은 Git을 설치한 후 터미널에서 실행합니다. 저장소를 ZIP으로 받았다면 압축을 풀고 해당 폴더로 이동한 뒤 가상 환경 생성부터 진행합니다.

```bash
git clone https://github.com/bhyungk/ECE6081-CourseProject.git
cd ECE6081-CourseProject
```

### macOS / Linux / WSL

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" jupyterlab
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" jupyterlab
```

PowerShell 예시는 가상 환경 활성화 없이 환경 안의 Python을 직접 실행합니다. `py -3.12`가 작동하지 않으면 Python 3.12를 설치하고 새 PowerShell 창을 엽니다. 이 설치 방법은 운영체제에 맞는 버전을 선택하므로 패키지 버전과 결과가 검증 환경과 조금 다를 수 있습니다. [고정 버전 목록](requirements-lock.txt)은 macOS arm64의 Python 3.12 검증 환경에서 생성되었습니다. 실행 환경의 버전을 보고서에 기록하세요.

## 기준 결과 확인과 실행

모든 명령은 저장소의 최상위 폴더에서 실행합니다. 아래 Windows 명령에서는 `python` 대신 `.\.venv\Scripts\python.exe`를 사용하세요.

### 1. 제공된 결과 확인 또는 재평가

`results/baseline/summary.csv`와 `participant_metrics.csv`를 바로 열어 볼 수 있습니다. 원본 데이터 없이 저장된 특징으로 기준 모델을 다시 평가하려면 다음을 실행합니다. 새 결과는 제공된 파일을 덮어쓰지 않도록 `results/recheck/`에 저장합니다.

```bash
python -m wesad_starter evaluate --features results/baseline/features.csv --output-dir results/recheck
python -m wesad_starter verify --output-dir results/recheck
```

### 2. 원본 신호부터 전체 기준 실험 실행

압축을 푼 데이터의 실제 경로로 `WESAD_ROOT`를 바꿉니다. `run`은 특징 추출, 15명 LOSO 평가, 결과 저장과 검증을 수행합니다. 이 예시는 기존 제공 결과와 분리해 `results/my_baseline/`에 저장합니다.

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

### 3. 설명 노트북 실행

노트북은 기본적으로 저장소에 포함된 `results/baseline/`의 결과를 읽고 저장된 특징을 이용해 LOSO 평가를 재실행합니다. 원본 신호를 직접 읽는 선택 예시를 실행하려면 `WESAD_ROOT`를 설정하고 `WESAD_DEMO_RAW=1`을 함께 설정하세요. `results/my_baseline/` 등 다른 결과를 읽으려면 실행 전에 `WESAD_RESULTS`를 해당 폴더로 설정합니다.

```bash
python -m jupyter lab notebooks/01_baseline_walkthrough.ipynb
```

Windows에서는 `python`을 `.\.venv\Scripts\python.exe`로 바꿉니다. JupyterLab에서 가상 환경의 Python 커널을 선택하고 셀을 위에서부터 실행하세요.

## 공통 실험 설정

- 입력: 손목 EDA 4 Hz, BVP 64 Hz / 원래 조건 라벨 700 Hz
- 구간: 기록 시작점 기준 60초 비중첩, 원래 조건이 일정한 1·2·3 구간만 사용
- 정답: Stress(2)=1, Baseline(1)·Amusement(3)=0 / 원래 조건 검사 후 이진 변환
- 기준 특징: 신호별 평균·표준편차·최솟값·최댓값·중앙값·IQR·기울기 7개
- 평가: 15명 LOSO, 기준 모델과 Advanced Model의 네 조건에서 동일 구간 사용
- 기준 전처리: 학습 참가자의 중앙값 대체 → StandardScaler / 전체 결측 열은 0 대체·유지
- 기준 모델: Logistic Regression, `C=1`, `class_weight="balanced"`, `max_iter=2000`, 임곗값 0.5
- 비교 조건: EDA, BVP, 특징 융합, EDA·BVP 확률의 0.5:0.5 평균 융합
- 주 지표: 참가자별 Macro-F1의 평균·표본 표준편차(`ddof=1`)
- 누수 방지: 메타데이터 입력 제외, 전처리·모델 선택에 외부 평가 참가자 정보 사용 금지

## 제공 코드와 직접 수행할 과제

이 저장소는 데이터 로딩·특징 계산·참가자 분할·Logistic Regression 평가·결과 저장과 RQ1의 출발점이 되는 참가자별 융합 차이 및 오류 중복 결과를 제공합니다. **Advanced Model과 RQ2~RQ4의 추가 실험은 각 팀이 직접 설계하고 구현해야 합니다.** 제공된 내부 분할 함수는 가중치 탐색까지 자동으로 수행하지 않습니다.

Advanced Model은 특징 추출, 모델 구조, 융합 방법 등을 자유롭게 설계할 수 있습니다. 머신러닝과 딥러닝 모두 사용할 수 있고, 특징 융합에는 신호별로 학습된 표현의 결합도 포함됩니다. 기준 모델과 차별화되는 개선 요소를 하나 이상 구현하고, 같은 참가자 분할과 평가 구간에서 기준 모델과 개발 모델 각각의 EDA·BVP·특징 융합·예측 융합을 비교하세요.

- **RQ1:** 참가자별 융합 이득·악화와 단일 신호 모델의 오류 중복을 분석합니다.
- **RQ2:** BVP 원신호에 사전 정의한 잡음 강도 두 수준을 적용하고 동일 구간의 손상 전후 EDA·BVP·융합 성능을 비교합니다. 잡음 종류·강도·적용 시점·재학습 여부와 난수 설정을 보고하고, 잡음을 적용한 신호에서 특징 또는 표현을 다시 계산합니다.
- **RQ3:** 예측 확률의 0.5:0.5 평균과 외부 학습 참가자 내부의 참가자 단위 검증으로 선택한 가중치를 비교합니다.
- **RQ4:** EDA 모델 2개의 앙상블, BVP 모델 2개의 앙상블, EDA 모델 1개와 BVP 모델 1개의 예측 융합을 비교합니다. 세 조건의 모델 수와 평가 참가자·구간을 같게 하고 학습 예산을 가능한 한 맞추며 차이를 기록합니다. 예를 들어 외부 학습 참가자에서 참가자 단위로 복원 추출해 두 학습 표본을 만들거나 확률적 초기화를 사용할 수 있습니다. 제공된 Logistic Regression은 seed만 바꿔 두 번 학습해도 예측이 같을 수 있으므로, 구성원의 예측 확률이 다른지 확인하고 구성 방법을 보고하세요. 외부 평가 참가자의 결과로 구성원을 선택하지 않습니다. 이 대조 실험은 앙상블 효과를 점검하지만 생리적 상호 보완성이나 인과관계를 입증하지는 않습니다.

### 내부 검증과 계산량

각 외부 LOSO 반복에서 한 참가자는 최종 평가용으로 남깁니다. 나머지 14명 안에서 참가자 단위 K-fold 또는 참가자 단위 검증 분할을 사용해 Advanced Model의 설정과 RQ3 융합 가중치를 선택할 수 있습니다. 한 참가자의 모든 구간은 내부 학습 또는 내부 검증 중 한쪽에만 들어가야 합니다. 각 내부 반복에서 결측값 대체·정규화 등 데이터에서 추정하는 전처리는 해당 내부 학습 참가자만으로 학습합니다. 설정을 정한 뒤 외부 학습 참가자 14명으로 다시 학습하고 남겨 둔 참가자에서 한 번 평가합니다. 외부 평가 참가자의 결과는 모델이나 가중치 선택에 사용하지 않습니다. 계산량이 크면 내부 fold 수와 탐색할 설정의 수를 줄이고 그 범위를 보고하세요.

## 결과

- 제공 결과: `results/baseline/`
- `window_manifest.csv`, `features.csv`: 구간 포함·제외 기록과 특징
- `predictions.csv`: 구간별 정답·확률·예측
- `participant_metrics.csv`, `summary.csv`: 참가자별 점수와 전체 요약
- `fusion_deltas.csv`, `condition_errors.csv`, `error_overlap.csv`: 융합 차이·조건별 오류·오류 중복
- `splits.json`, `preprocessing.csv`, `*_metadata.json`: 참가자 분할·전처리·실행 설정
- 기준 개수: 513구간(Baseline 282 / Stress 152 / Amusement 79), 예측 2,052행, 참가자 점수 60행
- 최종 제출: 본문 8쪽 이내 보고서(참고문헌·상세 설정 부록 별도), 실행 코드, 결과 CSV

## 참고

- [설명 노트북](notebooks/01_baseline_walkthrough.ipynb): 저장 결과 확인·평가 재실행
- [특징 명세](docs/feature_spec.md): 계산식·결측 처리
- [검증 기록](docs/validation.md): 실제 데이터 기준 결과·환경 재현
- 테스트 실행: `python -m pytest -q` (Windows: `.\.venv\Scripts\python.exe -m pytest -q`)
