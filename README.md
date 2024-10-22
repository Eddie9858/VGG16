# VGG16

![plot](/screenshot/VGG16_Structure.png)

### 소개

VGGNet은 2014년에 발표된 합성곱 신경망(ConvNet)으로, ImageNet 대규모 시각 인식 도전(ILSVRC)에서 높은 성능을 기록했습니다. 이 모델은 작은 3x3 필터를 깊게 쌓아 더 적은 파라미터로 기존 모델보다 더 높은 정확도를 달성했습니다.

### 네트워크 아키텍처

VGGNet은 16층과 19층의 신경망을 제안합니다. 이 네트워크는 여러 합성곱 층과 풀링 층으로 구성되며, 작은 3x3 필터를 사용합니다. 작은 필터를 사용하면 네트워크의 깊이를 증가시킬 수 있으며, 이는 특징 추출을 더 정교하게 만듭니다.

- **Conv 층**: 3x3 필터를 사용하여 특징 맵을 생성합니다.
- **Max-Pooling 층**: 2x2 맥스 풀링을 사용하여 차원을 줄입니다.
- **완전 연결(FC) 층**: 마지막에 3개의 완전 연결 층이 있으며, 각각 4096, 4096, 1000 노드를 가집니다. 1000 노드는 최종 분류를 위한 것입니다.
- **활성화 함수**: 모든 합성곱 및 완전 연결 층 뒤에는 ReLU 활성화 함수가 사용됩니다. ReLU는 비선형성을 더해주기 위해 사용되며, LRN(Local Response Normalization)은 이 구조에서 도움이 되지 않습니다.

기존의 모델은 11x11 필터(stride 4)나 7x7 필터(stride 2)를 사용했습니다. VGGNet에서는 다음과 같은 이유로 3x3 필터를 사용합니다:

- 동일한 receptive field를 가지면서 더 깊은 네트워크를 구성할 수 있습니다.
- 두 번의 3x3 필터를 거치면 receptive field는 5x5가 되고, 세 번 거치면 7x7이 됩니다.
- 세 번의 non-linear activation (ReLU)은 비선형성을 향상시켜 더 분별력 있는 결과를 얻게 합니다.
- 파라미터 수가 약 82% 감소합니다:
    - 3x3 필터의 파라미터 수: 27C²
    - 7x7 필터의 파라미터 수: 49C²
    
![plot](/screenshot/1.png)

### 학습

1. **훈련 방법**:
    - **미니 배치 경사 하강법**: 백프로파게이션을 사용하여 다중항 로지스틱 회귀 목적을 최적화합니다.
    - **배치 크기**: 256
    - **모멘텀**: 0.9
    - **정규화**:
        - 가중치 감소(L2 패널티 곱셈기: 5×10^-4)
        - 첫 두 개의 완전 연결 층에 드롭아웃 정규화(드롭아웃 비율: 0.5)
    - **학습률**: 처음에 10^-2로 설정, 검증 세트 정확도가 향상되지 않을 때 10분의 1로 감소, 총 3번 감소 후 370K 반복(74 에포크) 후 학습을 중단합니다.
2. **가중치 초기화**:
    - 깊은 네트워크의 그래디언트 불안정성을 해결하기 위해 랜덤 초기화로 얕은 구성 A를 처음에 훈련합니다.
    - 더 깊은 아키텍처 훈련 시, 첫 네 개의 Conv 층과 마지막 세 개의 완전 연결 층을 네트워크 A의 층으로 초기화하고, 중간 층은 무작위로 초기화합니다.
    - 무작위 초기화에서는 0 평균과 10^-2 분산을 가진 정규 분포에서 가중치를 샘플링하고, 바이어스는 0으로 초기화합니다.
3. **데이터 전처리**:
    - **고정 크기 224×224 이미지**: 훈련 이미지에서 무작위로 크롭합니다.
    - **데이터 증강**: 무작위 좌우 반전 및 RGB 색상 변화를 적용합니다.

### 훈련 이미지 크기 설정

1. **단일 스케일 훈련**:
    - **S=256**과 **S=384**에서 훈련된 모델을 평가합니다.
    - 처음에는 S=256을 사용하여 네트워크를 훈련합니다.
    - S=384 네트워크의 훈련 속도를 높이기 위해 S=256으로 사전 훈련된 가중치로 초기화하고 더 작은 초기 학습률(10^-3)을 사용합니다.
2. **다중 스케일 훈련**:
    - 각 훈련 이미지를 특정 범위 [Smin, Smax]에서 무작위로 S를 샘플링하여 재조정(Smin=256, Smax=512)합니다.
    - 스케일 지터링을 통해 훈련 세트를 증강하고 다중 스케일 이미지 통계를 더 잘 포착합니다.

### 테스트

테스트 단계에서는 훈련된 ConvNet과 입력 이미지를 사용하여 다음과 같이 분류를 수행합니다:

1. **이미지 재조정**: 이미지를 정의된 최소 이미지 크기(Q)로 등방성 재조정(isotropic scaling)합니다. Q는 훈련 스케일 S와 반드시 같을 필요는 없으며, 여러 Q 값을 사용하면 성능이 향상될 수 있습니다.
2. **네트워크 적용**:
    - 첫 번째 완전 연결 층을 7×7 컨볼루션 층으로, 마지막 두 개의 완전 연결 층을 1×1 컨볼루션 층으로 변환합니다.
    - 이 완전 컨볼루션 네트워크를 전체 이미지에 적용합니다.
    - 결과는 클래스 수와 같은 채널 수를 가지며 가변적인 공간 해상도를 갖는 클래스 점수 맵입니다.
3. **고정 크기 벡터 생성**: 클래스 점수 맵을 공간적으로 평균화하여 고정 크기 벡터의 클래스 점수를 얻습니다.
4. **데이터 증강**: 테스트 세트를 수평 반전 이미지로 증강합니다. 원본과 반전된 이미지의 소프트맥스 클래스 후행 확률을 평균화하여 최종 점수를 얻습니다.
5. **효율성**:
    - 완전 컨볼루션 네트워크를 사용하면 테스트 시 여러 크롭을 샘플링할 필요가 없으며, 이는 각 크롭마다 네트워크를 다시 계산해야 하므로 비효율적입니다.
    - Szegedy et al. (2014)처럼 많은 수의 크롭을 사용하는 것은 입력 이미지의 더 세밀한 샘플링을 제공하여 정확도를 향상시킬 수 있습니다.
    - 다중 크롭 평가와 밀집 평가는 컨볼루션 경계 조건이 다르기 때문에 서로 보완적입니다.

### 구현 세부 사항

우리의 구현은 공개된 C++ Caffe 툴박스(Jia, 2013)를 기반으로 하지만, 여러 GPU가 설치된 단일 시스템에서 훈련 및 평가를 수행할 수 있도록 많은 중요한 수정이 포함되어 있습니다.

1. **다중 GPU 훈련**:
    - 데이터 병렬성을 활용하여 훈련 이미지의 각 배치를 여러 GPU 배치로 나누어 각 GPU에서 병렬로 처리합니다.
    - GPU 배치 그래디언트가 계산된 후 이를 평균화하여 전체 배치의 그래디언트를 얻습니다.
    - 그래디언트 계산은 GPU 간 동기화되므로, 단일 GPU에서 훈련할 때와 동일한 결과를 제공합니다.
2. **속도 향상**:
    - 단순한 병렬 처리 방식을 통해 4-GPU 시스템에서 3.75배의 속도 향상을 달성했습니다.
    - 네 개의 NVIDIA Titan Black GPU가 장착된 시스템에서 단일 네트워크 훈련에는 아키텍처에 따라 2-3주가 소요되었습니다.

### 분류 실험

이 섹션에서는 ILSVRC-2012 데이터셋을 사용한 ConvNet 아키텍처의 이미지 분류 결과를 제시합니다.

### 단일 스케일 평가

1. **평가 설정**: 테스트 이미지 크기 Q는 S(고정된 스케일) 또는 0.5(Smin + Smax)(스케일 지터링의 경우)로 설정.
2. **LRN 효과**: LRN을 사용한 A-LRN 모델이 LRN을 사용하지 않은 A 모델보다 성능이 향상되지 않음.
3. **깊이에 따른 성능**: 깊이가 깊어질수록(11층에서 19층으로) 오류율이 감소.
    - 구성 C (1×1 Conv 층 포함)가 구성 D (3×3 Conv 층 사용)보다 성능이 낮음.
    - 스케일 지터링(S ∈ [256; 512])이 고정된 스케일보다 더 나은 결과를 제공.

![plot](/screenshot/2.png)

![plot](/screenshot/3.png)

### 다중 스케일 평가

1. **평가 방법**: 테스트 시 스케일 지터링을 통해 여러 재조정 버전에 대해 모델 실행 후 결과 클래스 후행 확률을 평균화.
2. **결과**: 스케일 지터링이 단일 스케일 평가보다 더 나은 성능을 제공.
    - 가장 깊은 구성(D와 E)이 가장 우수한 성능을 보임.
    - 구성 E는 테스트 세트에서 7.3%의 top-5 오류율 기록.

### 다중 크롭 평가

1. **비교**: 밀집 ConvNet 평가와 다중 크롭 평가 비교.
2. **결과**: 다중 크롭이 밀집 평가보다 성능이 약간 더 좋으며, 두 방법의 결합이 각각을 능가.
    - 경계 조건 처리가 다르기 때문.

![plot](/screenshot/4.png)

### ConvNet 결합

1. **평가 방법**: 여러 모델의 출력을 평균화하여 성능 향상.
2. **결과**: ILSVRC 제출 시 7개의 네트워크 앙상블로 7.3%의 테스트 오류 기록.
    - 제출 후, 두 모델을 결합하여 오류율을 7.0%로 감소.
    - 밀집 및 다중 크롭 평가를 결합하여 6.8%의 오류율 달성.

### Comparison

1. **결과**: ILSVRC-2014 챌린지에서 "VGG" 팀이 7.3%의 테스트 오류율로 2위.
    - 제출 후, 두 모델을 결합하여 오류율을 6.8%로 감소.
    - 매우 깊은 ConvNet이 이전 세대 모델을 크게 능가.
    - 단일 네트워크 성능에서는 GoogLeNet보다 0.9% 더 나은 7.0%의 테스트 오류 기록.

![plot](/screenshot/5.png)

![plot](/screenshot/6.png)

### **실험 결과**

VGGNet은 이미지넷 데이터셋에서 우수한 성능을 입증했습니다. VGG16과 VGG19 모델은 얕은 네트워크보다 더 나은 성능을 보였으며, 이는 깊이가 성능 향상에 중요한 요소임을 증명합니다. 특히, VGG16 모델은 단일 네트워크 성능에서 최고를 기록했습니다.

