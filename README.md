# [Team Task] Computer Vision 모델 조사 및 실험: MobileNet

본 리포지토리는 정확도 중심의 거대 CNN 모델들이 가진 한계를 실생활 서비스 관점에서 재해석하고, 경량화 모델인 **MobileNet**의 핵심 구조 분석 및 실험 결과를 담고 있습니다.

참고한 논문: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

논문 분석: [논문 리뷰](https://jangcarru20100919.tistory.com/49)

---

## 1. 모델 조사 (Model Investigation)

### 문제 의식 (Motivation)
* **CNN의 중량화**: ImageNet 대회 이후 CNN 모델들은 정확도를 높이기 위해 깊어지고 무거워졌습니다.
* **실행 환경의 한계**: GPU 환경에서는 구동 가능하나, 연산량($FLOPs$)과 파라미터 수의 급증으로 인해 모바일, IoT, 임베디드 기기에서는 사용이 불가능한 수준에 이르렀습니다.
* **핵심 질문**: "똑똑하지만 비싸고 무거운 모델을 어떻게 하면 실생활(모바일)에서 가볍고 빠르게 쓸 수 있을까?"

### 🛠 핵심 아이디어: "합성곱을 꼭 한 번에 해야 할까?"
기존 CNN(VGG, Inception 등)은 하나의 합성곱 연산에서 **공간 정보 추출**과 **채널 정보 결합**을 동시에 처리합니다. MobileNet은 이를 두 단계로 분리하는 **Depthwise Separable Convolution**을 제안합니다.



#### ① Depthwise Convolution (공간 특징 추출)
* 채널별로 독립적인 합성곱을 수행합니다.
* 채널 간 결합 없이 오직 **공간적 특징(선, 점 등)**만 추출합니다.
* *비유: 각 색깔 채널을 따로따로 살펴보며 밑그림을 그리는 단계*

#### ② Pointwise Convolution (채널 정보 결합)
* $1 \times 1$ 필터를 사용하여 앞선 결과들을 채널 방향으로 섞습니다.
* 여러 채널의 정보를 조합하여 **의미 있는 특징(클래스 판단 근거)**을 생성합니다.
* *비유: 개별적으로 그려진 밑그림들을 모아 최종적으로 무엇인지 판단하는 단계*

> **효율성 지표**
> 이 구조적 분리를 통해 기존 Standard Convolution 대비 연산량을 약 **8~9배 감소**시켰으며, 이는 곧 속도 향상과 배터리 소모 감소로 이어집니다.

### 모델 최적화 도구 (Hyperparameters)
MobileNet은 환경에 맞춰 모델 크기를 조절할 수 있는 두 가지 하이퍼파라미터를 제공합니다.
* **Width Multiplier ($\alpha$)**: 채널 수를 조절하여 모델의 두께를 결정합니다.
* **Resolution Multiplier ($\rho$)**: 입력 이미지의 해상도를 조절합니다.

---

## 2. 데이터셋 조사 (Dataset Investigation)

실험의 실무 적용 가능성을 검증하기 위해 아래 데이터를 활용했습니다.
* **데이터셋**: 육류 부위 분류 데이터셋 (Custom Public Dataset)
* **구성**: 총 5개 클래스 (소고기 등심·안심·채끝, 돼지고기 삼겹살·목살 등)
* **이미지 크기**: $224 \times 224$ (MobileNet 최적 규격)

---

## 3. 실험 설계 및 수행 (Experiment Design)

* **방법론**: 전이 학습(Transfer Learning) 및 2단계 파인 튜닝(Fine-tuning)
    * **Stage 1**: ImageNet 사전 학습 가중치를 고정하고 분류기(Classifier)만 학습
    * **Stage 2**: 전체 레이어의 동결을 해제하고 낮은 학습률($Learning\ Rate$)로 미세 조정
* **환경**: PyTorch / GPU T4 (Google Colab)

---

## 4. 결과 분석 (Result Analysis)

### 실험 결과 비교

| 모델 | 검증 정확도 (Accuracy) | 연산 효율성 | 비고 |
| :--- | :---: | :---: | :--- |
| **MobileNet v2** | **약 90%** | **최상 (Fast)** | 실시간 추론 및 모바일 적합 |
| **EfficientNet-B0** | 약 94% | 상 (Moderate) | 높은 정확도, 연산량 상대적 높음 |

### 최종 결론
* **성능 지표**: MobileNet은 정확도 면에서 최고 성능 모델은 아닐 수 있으나, **연산량 대비 성능(Efficiency)** 면에서 가장 압도적인 효율을 보입니다.
* **설계 방향의 전환**: 정확도 경쟁에만 매몰되지 않고, **"실제 사용 가능성"**에 초점을 맞춘 설계가 MobileNet의 핵심 가치임을 실험을 통해 확인했습니다.
* **적용 분야**: 실시간 객체 탐지, 모바일 앱 내 이미지 분류, 배터리 효율이 중요한 임베디드 기기 등.

---

## 📂 프로젝트 구조
* `1. MobileNet.ipynb`: MobileNet v2 학습 및 분석 코드
* `2. EfficientNet.ipynb`: 비교 분석용 대조군 코드
* `README.md`: 프로젝트 최종 리포트
