# WESAD 스트레스 조건 분류 시작 코드

- 제공 범위: 데이터 로딩, 특징 계산, 참가자 분할, Logistic Regression 평가, 결과 저장
- 학생 수행: Advanced Model 개발, 기준 모델과의 비교, RQ1~RQ4 실험·분석

## 설치

- 검증 환경: Python 3.12
- 데이터: 별도 준비한 WESAD의 동기화 `.pkl` 파일
- 폴더 구조: `$WESAD_ROOT/S2/S2.pkl` 등 15명(S2~S11, S13~S17)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

## 실행

- `WESAD_ROOT`: 본인의 WESAD 폴더 경로로 설정
- `run`: 특징 추출 → 전체 평가 → 결과 저장·검증
- `verify`: 저장된 예측·지표·분할의 일관성 재확인

```bash
export WESAD_ROOT="$HOME/ResearchDatasets/WESAD"
python -m wesad_starter run --data-root "$WESAD_ROOT" --output-dir results/baseline
python -m wesad_starter verify --output-dir results/baseline
```

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

## 학생 수행 사항

- Advanced Model: 개선 요소 1개 이상 구현, 기준 모델과 동일한 네 조건 비교
- RQ1: 참가자별 융합 이득·악화와 단일 신호 오류 중복 분석
- RQ2: BVP 잡음 두 수준 적용, 손상 전후 성능 비교
- RQ3: 동일 가중치와 학습 참가자 내부 검증으로 선택한 가중치 비교
- RQ4: 단일 신호 2모델 앙상블과 EDA+BVP 융합 비교, 모델 수·학습 예산 통제
- 최종 제출: 본문 8쪽 이내 보고서(참고문헌·상세 설정 부록 별도), 실행 코드, 결과 CSV

## 참고

- [설명 노트북](notebooks/01_baseline_walkthrough.ipynb): 저장 결과 확인·평가 재실행
- [특징 명세](docs/feature_spec.md): 계산식·결측 처리
- [검증 기록](docs/validation.md): 실제 데이터 기준 결과·환경 재현
- 테스트 실행: `python -m pytest -q`
