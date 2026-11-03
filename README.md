# WESAD 스트레스 조건 분류 시작 코드

대학원 프로젝트의 공통 실험을 재현하는 교수자·조교 제공 예제입니다. 동기화된 WESAD에서 손목 EDA와 BVP를 읽고, 구간 생성·특징 계산·참가자 단위 분할·Logistic Regression 평가·결과 저장을 수행합니다.

학생은 이 기준 모델을 출발점으로 **Advanced Model을 제안·구현하고 RQ1–RQ4를 모두 수행**해야 합니다. 예제 실행만으로 과제가 완료되지는 않습니다. 여기의 특징과 결과는 수업용 기준 설정이며, 기존 WESAD 논문 실험의 재현을 의미하지 않습니다.

## 설치와 전체 실행

검증 환경은 Python 3.12이며, 배포본의 `requirements-lock.txt`에 실제 검증한 패키지 버전을 기록합니다. 원본 WESAD는 이 배포물에 포함하지 않습니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps

export WESAD_ROOT="$HOME/ResearchDatasets/WESAD"
python -m wesad_starter run --data-root "$WESAD_ROOT" --output-dir results/baseline
python -m wesad_starter verify --output-dir results/baseline
```

`WESAD_ROOT` 바로 아래에 `S2/S2.pkl` 등 참가자 폴더가 있어야 합니다. 기본 실행은 S2–S11, S13–S17의 15명 전체를 사용합니다. S1과 S12는 원본 배포에서 센서 문제로 제외된 참가자입니다. 이미 압축이 풀린 동기화 `.pkl` 파일을 사용하며, 원본 E4 ZIP을 따로 풀거나 재동기화할 필요가 없습니다.

개발 도구를 포함하여 의존성을 새로 설치하려면 `python -m pip install -e '.[dev]'`를 사용할 수도 있습니다. 결과를 정확히 재현할 때는 검증한 lock 파일을 우선합니다.

## 코드 구성과 검증 기록

| 위치 | 역할 |
|---|---|
| `wesad_starter/data.py` | 데이터 로딩, 공통 시간 구간과 manifest 생성 |
| `wesad_starter/features.py` | 고정된 7개 특징과 신호별 입력 목록 |
| `wesad_starter/evaluation.py` | LR, LOSO, 내부 참가자 분할, 공통 결과 집계 |
| `wesad_starter/reporting.py` | CSV·설정 저장, 저장 결과와 추출 기록 검증 |
| `wesad_starter/cli.py` | 추출·평가·검증 명령 |
| `tests/` | 합성 입력을 이용한 규칙·누수·저장 결과 검증 |

배포 전 실제 데이터 실행과 별도 환경 재현 결과는 [검증 기록](docs/validation.md)에 있습니다. 노트북에서 생성하는 참가자별 그림은 `results/baseline/participant_macro_f1.png`에도 저장됩니다. 배포본에는 실제 기준 결과가 포함되어 있어, 원신호 재추출 전에도 설명 노트북을 실행하고 저장 특징으로 평가를 재현할 수 있습니다.

## 처리 단계별 실행

특징은 한 번 계산한 후 여러 모델의 실험에 재사용할 수 있습니다.

```bash
python -m wesad_starter extract --data-root "$WESAD_ROOT" --output-dir results/baseline
python -m wesad_starter evaluate --features results/baseline/features.csv --output-dir results/baseline
python -m wesad_starter verify --output-dir results/baseline
```

빠른 실행 확인에는 다음처럼 일부 참가자만 지정할 수 있습니다. 이 출력은 **pilot**이며, 최종 15명 LOSO 결과로 보고해서는 안 됩니다. 학습과 평가를 분리하려면 최소 두 참가자가 필요합니다.

```bash
python -m wesad_starter run --data-root "$WESAD_ROOT" --subjects S2 S3 --output-dir results/pilot
```

원본 데이터는 변경하지 않습니다. 큰 `.pkl`을 한 참가자씩 읽어 특징을 추출하며, 참가자 전체의 원신호를 동시에 메모리에 보관하지 않습니다. 출력 경로를 실험별로 구분하면 결과가 섞이는 것을 피할 수 있습니다.

## 데이터와 구간 규칙

| 항목 | 설정 |
|---|---|
| 입력 | `signal.wrist.EDA` 4 Hz, `signal.wrist.BVP` 64 Hz |
| 원래 조건 라벨 | `label` 700 Hz |
| 구간 | 기록 시간 0초부터 60초 길이, 60초 이동, 비중첩 |
| 구간별 표본 수 | EDA 240개, BVP 3,840개, 라벨 42,000개 |
| 유지 조건 | 원래 라벨이 구간 전체에서 동일하고 1·2·3 중 하나 |
| 이진 정답 | Stress(원래 2)=1, Baseline(1)·Amusement(3)=0 |
| 제외 | 조건 전환, 그 밖의 조건, 잘린 끝부분, 신호 길이 부족 |

**원래 조건의 순수성을 먼저 확인한 뒤 이진 라벨로 바꿉니다.** Baseline→Amusement 전환도 같은 non-stress로 합쳐서 유지하면 안 됩니다. 구간 ID는 참가자와 원래 시간 격자를 나타내며, 제외한 구간 때문에 뒤의 시간축이나 ID를 당기지 않습니다.

현재 확인한 원본 데이터의 기준 개수는 다음과 같습니다.

| 원래 조건 | 포함 구간 |
|---|---:|
| Baseline | 282 |
| Stress | 152 |
| Amusement | 79 |
| 합계 | **513** |

참가자별로 32–36개 구간이 유지됩니다. 완전한 60초 격자 중 전환을 포함한 223개와 순수한 제외 조건 706개는 제외됩니다. 잘린 마지막 구간도 manifest에 사유를 남깁니다. 데이터 버전이나 구간 규칙이 달라졌다면 기준 개수가 달라질 수 있으므로 그 이유를 먼저 확인해야 합니다.

## 특징과 기준 모델

각 신호에서 평균, 표준편차, 최솟값, 최댓값, 중앙값, IQR, 시간에 대한 선형 기울기 등 **7개 특징**을 계산합니다. EDA/BVP는 각각 7차원이며 특징 융합은 14차원입니다. 계산식과 실패 처리 규칙은 [특징 명세](docs/feature_spec.md)에 있습니다.

모델 입력은 명시적으로 지정한 `eda_`/`bvp_` 특징 열뿐입니다. 참가자 ID, 구간 ID, 시각, 원래 조건, 정답은 추적용 메타데이터이며 입력에 포함하지 않습니다.

각 외부 LOSO fold에서 한 참가자를 평가용으로 남기고, 나머지 14명에게만 다음 파이프라인을 맞춥니다.

1. 학습 특징의 중앙값으로 결측값 대체. 학습 자료에서 전체가 결측인 열은 0으로 대체하며 열을 유지합니다.
2. 학습 자료에만 `StandardScaler`를 맞추고 평가 자료에는 변환만 적용.
3. Logistic Regression: `C=1`, `class_weight="balanced"`, `max_iter=2000`, `solver="lbfgs"`, `random_state=42`.

| `input_mode` | 구성 |
|---|---|
| `eda` | EDA 7개 특징으로 LR 학습 |
| `bvp` | BVP 7개 특징으로 LR 학습 |
| `feature_fusion` | EDA+BVP 14개 특징으로 LR 학습 |
| `prediction_fusion` | 같은 fold에서 학습한 EDA·BVP 모델의 스트레스 확률을 0.5씩 평균 |

양성 클래스는 stress=1이며, 스트레스 확률이 **0.5 이상**이면 1로 예측합니다. 예측 융합을 위해 별도의 LR을 학습하지 않습니다. 특징 계산 실패를 이유로 특정 모델의 구간만 삭제하지 않으므로 네 조건의 평가 대상은 같습니다.

기준 예측에는 `decision_threshold=0.5`를 저장합니다. Advanced Model이 내부 검증으로 임곗값을 선택했다면, 공통 집계 함수에 전달할 예측 표의 `decision_threshold` 열에 fold·모델별 값을 기록할 수 있습니다. 이 열은 선택 절차를 대신 수행하지 않으며, 기준 LR의 임곗값은 계속 0.5로 고정합니다.

## 결과 파일과 해석

| 파일 | 내용 |
|---|---|
| `window_manifest.csv` | 모든 후보 구간의 메타데이터, 포함 여부와 제외 사유 |
| `features.csv` | 포함 구간의 메타데이터와 14개 특징 |
| `predictions.csv` | 외부 평가 구간별 모델 구성, 정답, 스트레스 확률, 예측 |
| `participant_metrics.csv` | 참가자·모델 구성별 Macro-F1과 구간 수 |
| `summary.csv` | 참가자별 Macro-F1의 평균과 참가자 간 표준편차 |
| `fusion_deltas.csv` | 두 융합 방식 각각의 EDA 대비·BVP 대비 참가자별 차이 |
| `condition_errors.csv` | Baseline·Amusement 오탐, Stress 누락의 분자·분모·비율 |
| `error_overlap.csv` | 참가자별 EDA·BVP 오류의 공통·단독 발생 개수 |
| `preprocessing.csv` | 각 외부 학습 fold의 대체값과 스케일러 설정 |
| `splits.json` | 외부 평가 참가자와 각 fold의 학습 참가자·구간 수 |
| `extraction_metadata.json` | 데이터 경로·원본 파일 정보, 구간·특징 규칙과 추출 환경 |
| `run_metadata.json` | 모델·평가 설정, 실행 명령, 패키지 버전과 출력 파일 해시 |

Macro-F1은 두 클래스 `[0, 1]`을 대상으로 계산하며 정의되지 않은 F1은 0으로 처리합니다. 주 결과는 **참가자별 점수를 같은 비중으로 평균한 값과 참가자 간 표준편차**입니다. 표준편차는 표본 표준편차(`ddof=1`)로 고정합니다. 전체 구간을 합친 점수를 주 결과로 대체하지 않습니다.

조건별 오류율에는 분자·분모를 함께 저장하며, 해당 조건의 구간이 없으면 비율은 0이 아닌 결측입니다. 융합 이득은 EDA 대비와 BVP 대비를 각각 제시합니다. 평가 결과를 보고 참가자마다 더 좋은 단일 신호 모델을 선택한 뒤 비교하지 않습니다.

전체 기준 실행에서는 예측 **2,052행(513구간×4조건)**, 참가자별 점수 **60행(15명×4조건)**이 생성됩니다. `verify`는 저장한 예측으로부터 점수·확률 평균·조건별 오류의 일관성을 확인합니다. 실제 성능 수치는 실행 후 생성되는 `summary.csv`를 사용합니다.

다음은 배포 전 실제 WESAD 15명 전체 실행과 결과 검수를 통과한 **현재 수업용 기준 결과**입니다. 위의 7개 통계 특징·고정 LR·60초 구간 규칙으로 계산한 참가자별 Macro-F1이며, 기존 논문이나 다른 전처리·특징·분할 설정과 같은 실험이라고 해석하지 않습니다.

| 입력 구성 | Macro-F1 평균 | 참가자 간 표준편차 |
|---|---:|---:|
| EDA | 0.643724 | 0.214955 |
| BVP | 0.728138 | 0.132157 |
| 특징 융합 | 0.782375 | 0.157720 |
| 동일 가중치 예측 융합 | 0.781896 | 0.149027 |

환경에 따라 부동소수점의 작은 차이가 생길 수 있습니다. 구간 수나 결과가 크게 다르면 데이터·특징·환경·분할을 확인하고, 위 수치에 맞추려고 평가 참가자를 이용해 설정을 조정하지 않습니다.

NaN·무한값·상수 구간 등의 검사는 기본 무결성 확인이며, 통과했다고 움직임 잡음이나 생리적 측정 품질이 보장되지는 않습니다. 구간 수 513개를 독립 참가자 수처럼 해석해서도 안 됩니다. 참가자는 15명이고 LOSO의 학습 집합은 서로 겹칩니다. 결과는 WESAD 실험 프로토콜로 정의한 조건 분류의 범위에서 설명해야 합니다.

## 설명 노트북과 개발 확인

`notebooks/01_baseline_walkthrough.ipynb`는 위 명령을 실행한 뒤 열어 사용하는 설명 노트북입니다. `.venv` Python을 커널로 선택합니다. 기본 실행은 저장된 실제 특징·결과를 읽고 같은 평가 모듈을 다시 실행하므로 큰 원신호를 모두 읽지 않습니다.

- 결과 위치를 바꾸려면 환경변수 `WESAD_RESULTS`에 출력 디렉터리를 지정합니다.
- 한 참가자의 원신호에서 특징을 직접 확인하려면 `WESAD_ROOT`를 지정하고 `WESAD_DEMO_RAW=1`로 설정합니다. 이 선택적 셀은 큰 참가자 `.pkl` 하나를 읽습니다.
- 코드의 핵심 검증은 `python -m pytest`로 실행합니다. 합성 입력을 사용한 단위 검증과 실제 WESAD의 기준 실험 결과는 구분합니다.

## 학생이 확장할 부분

시작 코드의 데이터 규칙·외부 참가자 분할·결과 형식은 공통으로 재사용하고, 모델과 특징을 확장할 수 있습니다. Advanced Model에서 학습으로 정하는 전처리·하이퍼파라미터·임곗값·융합 가중치는 외부 평가 참가자의 정보를 사용하면 안 됩니다. 내부 검증을 할 때에도 전처리는 해당 내부 학습 참가자에게만 맞춰야 합니다.

| 연구 질문 | 제공하는 지원 | 학생이 설계·구현·해석할 내용 |
|---|---|---|
| RQ1 | 참가자별 점수·융합 차이·오류 중복 분석의 공통 도구 | 개선·악화 참가자, 단일 신호 오류와 융합 이득의 관계 |
| RQ2 | 같은 구간과 결과 형식으로 재평가할 수 있는 구조 | BVP 잡음 두 수준, 적용 시점, 재학습 여부와 손상 효과 |
| RQ3 | 참가자 단위 내부 분할 예시 | 가중치 후보와 내부 검증·선택 절차, 동일 가중치와 비교 |
| RQ4 | 예측 확률을 결합하고 점수를 계산하는 공통 도구 | 실제로 서로 다른 예측을 내는 단일 신호 앙상블, 모델 수·학습 예산 통제 |

동일한 결정적 LR을 복제하거나 seed 이름만 바꾸어 같은 예측을 두 번 평균하는 것은 유효한 RQ4 대조가 아닙니다. 예제에는 Advanced Model이나 RQ2–RQ4의 완성 실험·답안을 포함하지 않습니다. 최종 보고서의 표와 그림은 제출한 예측·점수 CSV에서 추적할 수 있어야 합니다.

공통 확장 지점은 `wesad_starter.evaluation`의 `make_baseline`, `evaluate_loso`, `summarize_predictions`, `iter_inner_subject_splits`입니다. `summarize_predictions`에는 공통 예측 열을 유지하고 별도의 `model_family` 이름을 지정하여 학생 모델 결과를 전달할 수 있습니다. 참가자 내부 분할 예시는 다음과 같습니다. 이 예시는 분할만 만들며 가중치 후보나 선택 방법을 대신 정하지 않습니다.

```python
import pandas as pd
from wesad_starter.evaluation import iter_inner_subject_splits

features = pd.read_csv("results/baseline/features.csv")
outer_test_subject = "S2"
outer_train = features.loc[features["subject"] != outer_test_subject].reset_index(drop=True)
for train_idx, validation_idx in iter_inner_subject_splits(
    outer_train["subject"], outer_test_subject=outer_test_subject
):
    train = outer_train.iloc[train_idx]
    validation = outer_train.iloc[validation_idx]
    # 이 내부 train에만 전처리와 후보 모델을 fit하고 validation으로 선택합니다.
    assert set(train["subject"]).isdisjoint(validation["subject"])
    assert outer_test_subject not in set(train["subject"]) | set(validation["subject"])
```

참가자 분리와 학습 자료에서만 전처리를 맞추는 원칙은 scikit-learn의 [교차검증 안내](https://scikit-learn.org/stable/modules/cross_validation.html)와 [데이터 누수 방지 안내](https://scikit-learn.org/stable/common_pitfalls.html)에 설명되어 있습니다. 결측 열 유지 규칙은 [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)의 `keep_empty_features=True` 설정을 사용합니다.
