# Task 2 — Ultrasound-related VGG16 transfer candidate

이 브랜치는 아직 GPU 실행하지 않은 실험 후보입니다. `main`의 첫 VGG11, `experiment/mild-augmentation`, `experiment/robust-finetuning`은 보존합니다.

## 실행

`vgg.ipynb`는 코드 셀 하나입니다. Colab GPU 런타임에서 열고 `DATA_DIR`만 실제 Drive 데이터 폴더로 수정한 뒤 실행합니다. 기본 모델은 VGG16이며 VGG11도 설정값으로 지원합니다. 클래스 균형, 약한 label smoothing 0.03 및 증강 설정은 두 후보에서 동일합니다.

## Source/target 구분

[Comparison of fine-tuning strategies for transfer learning in medical image classification](https://arxiv.org/html/2406.10050v1)의 Table 1은 여러 선행연구를 정리한 표이며 source domain이 다양합니다. 논문 자체의 3.5절 실험 모델은 모두 ImageNet 사전학습을 사용합니다. 자체 실험의 target modality는 X-ray/MRI/조직병리/피부경/내시경이며 초음파는 포함되지 않습니다. 따라서 이 논문 결과를 초음파 최적 설정으로 간주하지 않습니다.

이번 코드의 실제 source는 **ImageNet**입니다. 공개 폐 초음파 VGG 가중치의 존재는 확인했지만 다운로드/전처리/구조 호환성을 검증하지 않아 이 코드에는 로드하지 않습니다. Source를 초음파로 바꾼 실험이라고 주장하지 않습니다.

## 가까운 연구와 적용 범위

| 연구 | 대상/방법 | 이 코드에 반영한 내용과 한계 |
|---|---|---|
| [Yang et al., ISBI 2019](https://arxiv.org/abs/1902.05582) | 돼지 심장 3D 초음파의 카테터 분할, pretrained VGG16 기반 DF-FCN, 전체 파라미터 Adam LR 1e-5, 회전/반전/대비/밝기 증강 | VGG16 후보와 작은 학습률의 전체 특징 적응을 검토. 원 연구의 FCN/U-Net decoder, 3D feature fusion, voxel masks는 사용하지 않음. 단일 프레임 분류 또는 새 인체 환자의 검증 연구가 아님. |
| [Mwikirize et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29512006/) | 동물 조직 2D 초음파의 바늘 검출/위치 추정, FCN+Fast R-CNN, 비의료 영상에서 전이 | 대상 형태와 modality가 가깝지만 detector/좌표 라벨을 필요로 하므로 구조를 그대로 사용하지 않음. |
| [Cheng & Malhi, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5359213/) | 복부 2D 초음파 11종 분류, ImageNet VGG16의 convolution을 고정하고 FC6/7/8 전체 재학습, study 단위 분리 | 출력층만 학습하던 준비 단계를 전체 기존 FC층의 적응으로 변경. 장기 분류와 sheath 판별은 다른 과제라 성능 우위를 보장하지 않음. |
| [Born et al., 2021](https://www.mdpi.com/2076-3417/11/2/672), [저자 코드](https://github.com/jannisborn/covid19_ultrasound) | 폐 초음파 VGG16 기반 분류와 CAM, 수정된 작은 classifier, 제한적 fine-tuning | CAM 진단과 patient/video 단위 검증 취지 참고. 기존 VGG FC 구조를 유지하며 폐 질환 가중치는 사용하지 않음. |

## 코드 변경

1. 원래 VGG16의 13 convolution + 기존 3 FC 구조를 유지하고 출력만 2 클래스로 교체합니다.
2. 동결된 VGG convolution+pool5 특징을 한 번 추출합니다. 이는 동일 VGG 자체이며 다른 신경망이 아닙니다. 이 특징으로 기존 FC층 전체를 3 epoch 준비 학습합니다.
3. 공통 가중치에서 두 후보를 독립적으로 학습합니다. A `vgg_late2_fc`: 뒤쪽 두 합성곱 블록+전체 FC. B `vgg_full`: 전체 층. 원래 두 후보의 환자별 손실 비교는 이번 실험에서 제외하고 미세조정 범위만 비교합니다.
4. 앞쪽 conv LR 2e-6, 뒤쪽 conv LR 1e-5, 기존 FC LR 5e-6, 출력 LR 5e-5. 값은 본 과제용 선택이며 논문의 최적값을 재현한다는 뜻이 아닙니다.
5. 같은 검증 환자 G06/G07/G09/G14/G17/G20, 599장과 train 2349장을 유지합니다. 임계값 조정 BA 우선, AUROC 동률 기준입니다. 후보/epoch/임계값을 같은 Validation으로 선택하므로 독립 Test 점수가 아닙니다.
6. 선택 모델의 검증 환자별 원본과 positive-logit Grad-CAM을 함께 저장합니다. CAM은 진단용이며 정답 위치를 증명하거나 분류에 사용하는 별도 모델이 아닙니다. 자동 ROI/crop을 적용하지 않습니다.

## 시간/결과

두 후보의 미세조정에 합계 기본 35분을 배분하고 각각 최대 12 epoch, patience 4입니다. VGG16은 VGG11보다 합성곱 계산이 많습니다. 데이터 캐시/모델 다운로드/FC 준비/초기 평가/최종 저장/Grad-CAM 시간은 별도이므로 전체 35분 완료를 보장하지 않습니다.

가중치와 추론 코드/학습 코드/패키지 버전을 Drive `task2_vgg_ultrasound/<시간>`에 저장하고 작은 `vgg_ultrasound_report.zip`만 다운로드합니다. 최고 가중치는 3 epoch마다 및 후보 종료 시 Drive에 복사하며 자동 재개는 지원하지 않습니다. 앙상블, TTA, 전체 데이터 재학습은 없습니다.

첫 VGG11의 BA@0.5=0.740966, AUROC=0.826996입니다. 첫 모델의 저장 예측에 같은 정확한 ROC 임계값 탐색을 적용하면 BA=0.758088이며 이 값을 비교선으로 사용합니다. 단순 임계값 재조정은 새 학습의 성능 개선이 아닙니다.

코드 문법, 학습/추론 아키텍처 일치, FC 적응 범위, convolution 블록 경계, 같은 손실 유지, CAM hook 순서, 동점 점수를 포함한 100개 ROC/AUC 계산 사례를 확인했습니다. 이 작성 환경에는 torch가 없어 GPU 학습 및 추론 전체 실행은 검증하지 못했습니다. 논문 성능을 본 데이터의 예상 성능으로 제시하지 않습니다.
