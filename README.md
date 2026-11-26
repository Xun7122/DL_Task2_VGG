# Task 2 — Pretrained VGG11

[Colab에서 한 셀로 실행](https://colab.research.google.com/github/Xun7122/DL_Task2_VGG/blob/main/vgg.ipynb)

초음파 영상에서 sheath 유무를 분류하는 VGG11 기준 실험입니다. `main`에 첫 실험의 학습 설정을 보존하고, 개선 후보는 별도 브랜치에서 비교합니다.

## 실행

1. 위 Colab 링크를 열고 GPU 런타임을 선택합니다.
2. Drive에 `labels.csv`와 이미지 폴더가 포함된 데이터 폴더를 준비합니다.
3. 유일한 코드 셀 맨 위 `DATA_DIR`을 실제 Drive 폴더로 수정하고 실행합니다.

`labels.csv`에는 `sample_id,path,label,patient_group`이 필요합니다. 이미지 경로는 CSV가 있는 폴더 기준입니다. 폴더 아래 CSV가 한 단계 더 들어 있어도 자동 탐색합니다. 이미지 파일은 Colab 로컬로 복사한 뒤 학습합니다.

가중치 `vgg.pt`, 실제 패키지 버전, 학습 이력 및 예측 결과는 Drive의 `task2_vgg_results`에 저장합니다. 완료 후 작은 분석 보고서 ZIP만 다운로드합니다. 중단 후 자동 재개 기능은 없으며 무료 Colab의 연결 유지와 실행 시간은 보장되지 않습니다.

## 기준 설정

| 항목 | 설정 |
|---|---|
| 모델 | torchvision VGG11, ImageNet-1K 사전학습 |
| 출력 | 마지막 FC층을 2개 클래스로 교체 |
| 입력 | 높이 336 × 너비 224, ImageNet 정규화 |
| 증강 | 좌우 반전 0.5, 회전 ±5도 |
| 배치 | 8 |
| 손실 | 학습 클래스 빈도의 역수로 가중한 CrossEntropy |
| 1단계 | 출력층만 3 epochs, AdamW LR 1e-3 |
| 2단계 | 전체 층 최대 25 epochs, 특징/기존 FC LR 1e-5, 출력 LR 1e-4 |
| 정규화 | weight decay 1e-4, VGG 기본 dropout |
| 종료/선택 | patience 8, BA@0.5 우선·동률 AUROC |
| 임계값 | 최선 가중치 선택 후 0.10~0.90, 간격 0.01 탐색 |

환자 단위 StratifiedGroupKFold(5 folds, seed 42)의 fold 0을 검증에 사용합니다. Train은 21명/2349장, Validation은 **G06, G07, G09, G14, G17, G20**의 6명/599장입니다. 환자가 양쪽에 겹치지 않는지 코드에서 검사합니다.

## 첫 실행 기록

| 지표 | Validation |
|---|---:|
| BA @ 0.5 | 0.740966 |
| 임계값 조정 BA | 0.755725 |
| AUROC | 0.826996 |
| 선택 epoch | 8 |
| 선택 임계값 | 0.10 |

사용자가 제공한 `vgg_report.zip`에서 가져온 기존 실행 결과입니다. 이 저장소를 올리면서 재학습한 결과가 아닙니다. 저장된 첫 코드에서 데이터 입력 부분만 Drive 폴더 방식으로 정리하고 기준 분할 검사를 추가했습니다. 라이브러리와 GPU 환경에 따라 재실행 수치는 달라질 수 있습니다.

동일한 Validation을 epoch와 임계값 선택에 사용했으므로 **독립 Test 성능이 아닙니다**. 전체 데이터 재학습이나 앙상블은 포함하지 않습니다. 원본 데이터와 가중치는 저장소에 포함하지 않습니다.

## 브랜치 운영

- `main`: 첫 VGG 기준 코드와 결과를 보존합니다.
- `experiment/mild-augmentation`: 첫 설정에서 밝기·대비 ±10% 증강만 추가합니다.
- 후속 실험은 각각 `main` 또는 검증된 후보에서 분기하고 변경점 하나씩 비교합니다.

실험 결과는 `experiment_log.csv`에 기록합니다. 같은 검증 환자로 비교한 뒤 최종 후보를 다른 환자 fold에서도 확인합니다.
