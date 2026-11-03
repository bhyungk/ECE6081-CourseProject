# 배포 전 검증 기록

검증일: 2026-11-03 (한국 시간). 이 기록의 실제 데이터 결과는 동기화된 WESAD 15명 전체에서 생성했습니다. 단위 테스트의 합성 입력으로 얻은 점수와 구분합니다.

## 실제 데이터 실행

- 대상: S2–S11, S13–S17, 손목 EDA 4 Hz / BVP 64 Hz / 라벨 700 Hz.
- 기록 시작점 기준 60초 비중첩 구간, 원래 조건의 순수성 검사 후 이진 매핑.
- 포함 구간 513개: Baseline 282 / Stress 152 / Amusement 79.
- 15명 LOSO, 3개 LR을 fold마다 학습하고 두 단일 신호 확률을 재사용하여 네 조건 평가.
- 구간별 예측 2,052행, 참가자별 점수 60행.
- 원본 데이터 파일은 읽기만 수행했으며 배포 ZIP에 포함하지 않음.

| 입력 구성 | 참가자 Macro-F1 평균 | 표본 표준편차 |
|---|---:|---:|
| EDA | 0.643724 | 0.214955 |
| BVP | 0.728138 | 0.132157 |
| 특징 융합 | 0.782375 | 0.157720 |
| 예측 융합 | 0.781896 | 0.149027 |

기준 결과는 `results/baseline/`에 저장합니다. 특징/모델 설정은 외부 평가 결과에 맞춰 조정하지 않았습니다.

## 코드·저장 결과 검수

- `python -m pytest -q`: **36개 통과**. 구간 전환·시간 대응·잘린 신호·특징 계산·비정상 입력, 학습 전처리 분리, 전체 결측 열 유지, 메타데이터 차단, 참가자 분할, 지표와 결과 저장을 검증.
- `python -m wesad_starter verify --output-dir results/baseline`: 통과. 파일 해시, 추출 기록과 입력의 일치, LOSO 평가 범위, 동일 구간, 학습 참가자만 사용한 대체값·스케일, 확률 평균, 지표·차이·오류 재계산, manifest 포함 여부를 검사.
- 설명 노트북 13셀을 실제 저장 결과로 실행했고 코드 셀 오류 없음. 참가자별 Macro-F1 그림도 확인.
- 독립 검토에서 오래된 추출 기록이 새 특징과 섞일 수 있는 문제를 수정하고 회귀 검증 추가.

## 별도 새 환경 재현

새 임시 Python 3.12 환경에서 README 순서대로 실행했습니다.

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip check
python -m pytest -q
python -m wesad_starter evaluate --features results/baseline/features.csv --output-dir /tmp/wesad-recheck
python -m wesad_starter verify --output-dir /tmp/wesad-recheck
```

고정 버전 56개 일치, 의존성 검사·36개 테스트·평가·결과 검증을 통과했습니다. 원신호는 최초 전체 실행에서 추출했고, 별도 환경에서는 저장 특징을 재사용했습니다. 두 환경의 특징, manifest, 예측, 참가자 지표, 요약, 융합 차이, 조건 오류, 오류 중복, 전처리 통계 CSV 9개가 모두 바이트 단위로 일치했습니다.

검증 환경: Python 3.12.14, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, SciPy 1.18.1, macOS. 전체 버전은 lock 파일과 실행 메타데이터에 남깁니다. 다른 OS에서의 실행은 별도로 검증하지 않았습니다.

이 검수는 구현·평가·저장 결과의 일관성을 확인합니다. 생리적 측정 품질이나 다른 데이터셋에 대한 일반화 성능의 보증은 아닙니다.
