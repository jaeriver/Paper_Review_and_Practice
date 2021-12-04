## 논문 파일

[morphling-socc21.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4f7182a3-de27-4137-9956-05afba4d90c2/morphling-socc21.pdf)

## 공부해야할 사전 지식

- Bayesian optimization, linear regression, transfer learning
- meta learning
[https://rhcsky.tistory.com/5](https://rhcsky.tistory.com/5)
- few-shot learning problem
[http://dmqm.korea.ac.kr/activity/seminar/301](http://dmqm.korea.ac.kr/activity/seminar/301)
- Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
    
    [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/573ea66b-753d-4fcb-b2cf-c829e4112fb1/Model-Agnostic_Meta-Learning_for_Fast_Adaptation_of_Deep_Networks.pdf)
    
- (완료)Mapping function (regression tasks)
- exploration-exploitation trade-off

## 0. Abstract

meta-learning 을 이용해서 많은 구성요소 중 최적화된 모델을 찾는다. Alibaba와 비교.

## 1. Introduction

- hardware configuration
    
    CPU cores, GPU types, GPU memory, GPU share(if gpu sharing is supported)
    
- Alibaba
    
    low resource utilization ( 80%의 cpu, gpu 메모리 할당량을 안씀)
    
- Prevalent auto-configuration techniques
    
    Bayesian optimization, linear regression, transfer learning
    
    저차원 구성 공간에서는 클라우드 작업 부하를 잘 조정하지만 작업 공간이 늘어나면 최적점을 찾기위한 large sampling overhead가 증가하여 비효율적이다.
    
- 다양한 모델임에도 similiar tendency가 있음 따라서 few-shot learning problem을 공식화하고 MAML 기술을 사용하여 다양한 하드웨어와 런타임 구성 환경에서 성능이 어떻게 변하는지 포착하고자 함.

## 2. Background

## 3. The Need for Configuration Tuning

### 3.3 Prior Art and Their Inefficiency

### Auto-Configuration Using Search

자동 구성에 또다른 접근은 샘플링을 작은 개수로 만드는 것임. 성능 평가를 위해. 세가지 알고리즘이 있음.

- Black-box search : sequential model-based optimization (SMBO)
    
    가장 최적의 구성을 찾기 위함. 검색 프로세스가 진행되는 동안 SMBO는 회귀 모델을 빌드함(가우시안 등) configuration-performance 곡선에 맞추기 위해 진행됨. 샘플링 예산이 소진될 때 까지 다음의 구성을 무한 루프를 돌며 샘플링함
    
    SMBO중 가장 유명한 것은 베이지안 optimization임. 하지만 저차원 검색에서는 효율적인 반면 큰 고차원 구성 공간을 탐색하는데 매우 비쌈. 따라서 캡처 6에서 추론 서비스에 베이지안이 비효율적임을 보일 에정.
    
- White-box prediction
    
    특정 구성에서 성능을 예측하고 이를 사용하여 검색 프로세서를 구동하는 접근 방식.
    
    구성에 따라서 성능이 어떻게 변경될 수 있는지에 대한 사전 지식을 사용하여 몇 가지 샘플링으로 회귀 모델을 구축하는 것이 특징. 하지만 고차원의 configuration-performance 평면은 너무 복잡하여 몇 가지 샘플링으로는 거의 적합하지 않음
    
- Similarity-based search(유사성 기반 검색)
    
    각 tuning과 벤치마킹 워크로드 사이의 유사성을 측정하고 이를 사용하여 검색 프로세스를 안내함
    
    주로 현재 작업 부하와 이전에 연구된 벤치마크 간의 일대일 유사성에 초점을 맞춤. 세션6에서 유사성 기반 검색은 쵲거의 구성을 찾기 위해 큰 구성 공간을 샘플링 해야 함.
    

## 4. Algorithm Design

### 4.1 Common Performance Trend

RPS(Request Per Second)가 유형의 리소스를 더 많이 구성하여 향상되지만 병목 현상이 다른 리소스 (GPU memory to CPU cores)로 이동함에 따라 성능 향상이 감소하는 경향이 있음을 관찰.

GPU Memory는 최소 요구 조건에 해당됨, 예를들어 높은 batch의 경우 GPU Memory가 부족할 경우 아예 작업 수행이 되지 않음

리소스, 런타임 구성은 다른 ML모델을 실행하는 다양한 추론 서비스에 일반적인 성능 영향을 미침.

Resource Configuration :  GPU Mem, CPU Cores

- 대규모 모델을 로드하거나 대규모 요청 배치를 제공하려면 대용량 GPU 메모리 필요
- CPU코어가 많을수록 성능이 약간 향상됨

Runtime Parameters: batch size

각 모델마다 구성 별로 성능 전환점이 존재하고 이러한 동기로 메타 학습 기술을 사용하여 빠른 자동 구성 접근 방식을 유도. 일반 추론 서비스에 대해 공통 구성 성능 추세를 포착하는 메타 모델을 오프라인으로 훈련함.

### 4.2 Meta-Model Training

### Few-shot Regression

- 공식 이해
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1854a0db-9f8b-4b9f-b527-88091d1d9ccb/Untitled.png)
    

K개의 샘플 데이터는 학습 오버헤드를 최소화 하기 위해 (전체 space의 5% 정도를 차지) 굉장히 작은 숫자가 될 것이고 이것이 한계이다.

### Model-Agnostic Meta-Learning (MAML)

K-shot 회귀의 약점을 보완하기 위함. 두 가지의 stage로 회귀 모델을 학습시킴

- Stage-1 : Meta-Model Training
    
    set of regression tasks를 
    
    기능 학습의 관점에서 메타 모델 훈련은 기본적으로 많은 관련 작업에 광범위하게 적용할 수 있는 국가 간 표현을 구축한다. 메타 모델은 나중에 SGD를 사용하는 새로운 회귀 작업에 적응하므로 나중에 SGD 프로세스가 과도하게 적합되지 않고 빠르게 진행될 수 있도록 훈련해야 한다. 따라서 메타 모델 𝛽는 T에서 샘플링한 과제, 즉 ←←𝜃𝑚-𝛽 L(𝑓)에 대해 최적화하도록 훈련된다. 여기서 is는 메타 훈련 단계의 학습 단계이고 𝜃i는 Eq(3) 이후의 초당 d 단계에서 계산된 과제 𝑇𝑖에 대한 미세 조정 모델이다. 알고리즘 1은 메타 모델의 훈련 과정을 자세히 설명합니다.
    
- Stage-2 : Fast Adaptation
    
    메타 모델 𝜃𝑚이 학습되면 새 작업 𝑇𝑖에 대한 초기 회귀 모델로 사용되며 𝑇𝑖에 더 잘 맞도록 미세 조정 프로세스가 뒤따릅니다(Eq. (3) 참조). 메타 훈련은 빠른 적응을 가능하게 하기 위한 것이기 때문에 이러한 미세 조정은 몇 개의 데이터 포인트로 빠르게 수렴됩니다. 이는 매개 변수의 작은 변경과 같이 작업의 변경에 민감한 메타 모델 𝜃𝑚을 찾는 것을 목표로 합니다. 모든 𝑇𝑖 의 손실을 크게 개선할 것입니다. 다음으로 빠른 적응과 함께 훈련된 메타 모델을 사용하여 최적의 구성을 검색하도록 지시하는 새로운 SMBO(순차적 모델 기반 최적화) 접근 방식을 개발합니다.
    

### 4.3 Directing SMBO Search with Meta-Model

학습된 meta-model을 가지고 최적의 구성을 찾기 위해 SMBO를 이용함.

- Standard form
    
    SMBO(Sequential Model-Based Optimization) 작업을 수행.
    
    회귀 모델에 대해서 랜덤하게 초기화 → 모델 fitting과 그 모델을 쓸지 말지 결정하는 과정을 반복함
    회귀 검색은 표본 추출 예산(특정된 예산)이 부족할 때 까지 진행함. 탐사와 착취(exploitation) 사이에서 균형을 잡는 것이 중요한데 획득 함수가 회귀 모델에 의해 만들어진 예측의 평균과 분산을 모두 결합해서 절충을 탐색하기 위해 정의됨
    
- Meta-Model as an Initial Regression Model
    
    Standard form과 달리 학습된 meta-model인 𝜃𝑚를 처음에 사용함, 서치 과정 동안 새로운 inference service(Ti에 의해 회귀 작업이 된)에 적응을 함
    
- Exploration-Exploitation Trade-off
    
    만약 알고리즘 exploitation만 한다고 가정하면, 항상 높은 예측으로 구성을 샘플링 → 검색이 로컬 환경에 최적 상태로 고정되게 쉬움
    따라서 exploration(검색)과 exploitation(부당한 사용) 사이의 균형을 맞추는 것이 중요함
    그러나 예측의 불확실성에 대해 알기가 어렵고 prediction confidence(예측 신뢰도)라는 것을 정의하여 이 문제를 해결함
    

### 4.4 Why do we use Meta-Learning?

메타 학습에서 오프라인으로 훈련된 메타 모델은 추론 서비스의 일반적인 기능, 일반적인 성능 추세를 자동으로 포착함.

configuration search를 할 때 정보와 오버피팅이 되지 않은 사전 정보를 제공하며 새로운 추론 서비스에 몇번의 few shots을 통해 새로운 추론 서비스에 정확히 맞출 수 있음.

따라서 블랙박스, 화이트 박스 예측의 이점이 결합됨.

## 5. Cloud-Native Implementation

Kubernetes에서 Golang Code 기반으로 배포됨

### Programming Interface

사용자 입장에서 설정해 주어야 할 것들이 나열됨

### Workflow

## 6. Evaluation

이 섹션에서는 AWS EC2의 인기 있는 오픈 소스 모델과 프로덕션 클러스터에서 실행되는 실제 추론 서비스로 Morphling을 평가합니다. 우리의 평가는 세 가지 질문에 답하는 것을 목표로 합니다. (1) 구성 최적성 및 검색 비용(§6.1.2 및 §6.2) 측면에서 기존 자동 튜닝 솔루션과 비교하여 Morphling의 성능은 어떻습니까? (2) Morphling이 다양한 성능 목표를 지원할 수 있습니까(§6.1.2)? (3) Morphling은 어떻게 새로운 구성 작업에 빠르게 적응합니까(§6.1.3)?

### 6.1.1 Methodology

오픈 소스 모델. MLPerf Inference 벤치마크[45]의 지침에 따라 ResNet[31, 32], EfficientNet[59] 및 MobileNet[ 34, 53] 및 BERT[23], ALBERT[42] 및 Universal Sentence Encoder[20]와 같은 언어 모델이 있습니다. 이러한 사전 훈련된 모델은 TensorFlow 모델 동물원에서 제공합니다[8, 15]. 컨테이너 시작 시 리소스와 배치 크기를 구성하기 위한 인터페이스와 함께 모델과 서빙 프레임워크를 Docker 컨테이너[3]에 패키징합니다.
검색 공간. 모델 제공 컨테이너에 대해 (1) CPU 코어, (2) GPU 메모리(총 용량의 백분율), (3) 요청 배치 크기, (4) GPU 유형의 4가지 조정 가능한 구성 노브를 고려합니다. 각 구성 노브에 대해 오프라인 측정을 수행하여 검색 공간을 결정합니다. 예를 들어, 5개 이상의 CPU 코어가 있는 구성은 추론 RPS를 더 이상 개선할 수 없기 때문에 고려하지 않습니다. 표 3은 각 구성 노브에 대한 가능한 선택 사항을 요약한 것입니다. 함께 검색 공간에는 총 720개의 구성 옵션이 있습니다. 이는 기존의 클라우드 구성 작업과 비교할 때 큰 것으로 간주됩니다. 예를 들어 Cherrypick[18] 및 Scout[36]에서 연구한 VM 구성의 검색 공간에는 선택 항목이 수십 개에 불과합니다.
목적. 우리는 구성 조정의 목표를 금전적 비용당 서비스 처리량을 최대화하는 것으로 설정했습니다.
최대𝑥 ∈ A RPS/비용. (7)
특히, 실험에서 1초로 설정된 대기 시간 SLO에 따라 컨테이너의 최대 RPS를 스트레스 테스트합니다. 모델 제공의 금전적 비용을 측정하기 위해 비용 모델을 가정합니다. 비용 = 기본 비용 + GPU 가격 × GPU 메모리 + CPU 가격 × CPU 코어 수. 표 2는 기본 비용 = 0.2 USD, CPU 가격 = 0.02 USD, T4 가격 = 0.4 USD, M60 가격 = 0.4 및 V100 가격 = 2.6 USD.
모플링 설정. Morphling에 사용된 메타 모델은 각각 128개의 은닉 유닛이 있는 2개의 은닉 레이어가 있는 신경망입니다. 42개의 ML 모델 중 10개는 메타 학습용으로, 나머지는 테스트용으로 사용합니다. 공정하고 재현 가능한 비교를 위해 CPU 코어, GPU 메모리 및 배치 크기의 최대값과 최소값을 포함하여 초기 샘플링 지점으로 8개의 고정 구성을 선택합니다. 우리는 실험에서 T4 GPU를 사용합니다. 우리는 이러한 구성을 초기 지점으로 샘플링하는 것이 기본 알고리즘, 특히 BO에 대한 최상의 성능으로 이어지는 반면 Morphling은 초기 선택에 둔감하다는 것을 발견했습니다.

기준선. 자동 구성을 위한 5가지 기준선 알고리즘과 Morphling을 비교합니다.

1. 베이지안 최적화(BO): Cherrypick[18]과 유사하게, 우리는 획득 함수로 신뢰 상한을 갖는 가우스 회귀자를 사용합니다.
2. Ernest[61]는 각 작업 부하에 대한 전용 회귀 모델을 구축하고 몇 가지 샘플링으로 이를 훈련합니다. Morphling에서와 동일한 신경망 아키텍처를 사용하지만 처음부터 각 추론 서비스에 대해 학습합니다.
3. Google Vizier[27]는 가우스 회귀자를 커널 기능으로 사용하고 잘 프로파일링된 벤치마크로 새로운 검색을 가속화하기 위해 전이 학습을 사용하는 유사성 기반 검색 프레임워크입니다. 10개의 ML 모델을 오프라인 벤치마크로 사용합니다. 그런 다음 각 테스트 모델은 벤치마크의 데이터와 테스트 벤치마크 잔차에 맞는 가우스 회귀자로 표시됩니다.
4. Fine-Tuning은 또 다른 간단하면서도 효과적인 유사성 기반 검색 접근 방식입니다. Morphling과 유사하게 10개의 ML 모델을 사용하여 회귀 모델을 오프라인으로 학습합니다. 그러나 목표는 미래의 빠른 적응을 고려하지 않고 단순히 평균 예측 정확도를 향상시키는 것입니다. 그런 다음 훈련된 모델을 새 서비스 모델로 개선합니다.
5. 랜덤 검색은 검색 공간을 무작위로 샘플링하여 다양한 구성을 생성하고 가장 성능이 좋은 것을 선택합니다. 우리의 구현에서 동일한 구성은 한 번만 샘플링됩니다. 즉, 교체 없는 무작위 검색입니다.

측정항목. 평가에 두 가지 측정항목을 사용합니다. (1) Eq.의 목적 함수에 의해 정의된 선택된 구성에서 결과적인 성능. (7). 철저한 검색을 통해 찾은 최적 구성의 성능에 대해 정규화된 값을 보고합니다. (2) 특정 성능 요구 사항(예: 최적의 70%)을 충족하는 구성을 찾는 데 필요한 샘플링 수로 측정한 검색 비용.

### 6.1.2 Performance, Cost, and Generality

검색 품질 및 검색 비용. Morphling과 5가지 기본 접근 방식을 사용하여 32개 테스트 모델 모두의 구성을 조정합니다. 그림 6a는 다양한 샘플링 예산에서 식별된 구성의 정규화된 성능을 비교합니다. 여기서 상자는 모든 테스트 모델 성능의 25번째, 50번째, 75번째 백분위수를 나타내고 수염은 10번째 및 90번째 백분위수를 나타냅니다. 동일한 샘플링 예산으로 Morphling은 항상 5가지 기준보다 더 나은 구성을 반환합니다. 실제로 Morphling은 30개 이하의 구성(검색 공간의 5% 미만)을 샘플링하여 모든 모델에 대한 최적의 구성을 식별합니다. 두 번째로 좋은 알고리즘인 Fine-Tuning은 200개의 구성을 샘플링해야 하지만 여전히 모든 모델에 대해 최적의 성능을 보장할 수는 없습니다.

그림 6b는 특정 성능 요구 사항을 충족하기 위해 다양한 접근 방식에 필요한 검색 비용을 더 비교합니다. 여기서 막대는 10번째 백분위수와 90번째 백분위수로 확장된 오차 막대로 32개 모델을 조정하는 데 드는 중앙값 비용을 측정합니다. 모든 경우에 Morphling은 다른 기준을 능가합니다. 성능 요구 사항이 높을수록 더 효율적입니다. 특히 최적 구성(100% 최적성)을 검색할 때 Morphling은 Fine-Tuning(54개 샘플링의 중앙값 필요)보다 3배, BO 및 Google Vizier보다 9.4배, 22배 이상 효율적입니다. 어니스트 및 무작위 검색. Ernest는 VM 선택[61]과 같은 작은 구성 공간으로 더 간단한 클라우드 자동 조정 문제를 해결하는 데 효율적이기는 하지만 모델 제공에 의해 요구되는 고차원 검색을 탐색하는 데는 부족하다는 점을 언급할 가치가 있습니다.

다양한 목적 기능 지원. Morphling의 고성능 및 저렴한 비용은 특정 목적 함수에 얽매이지 않고 일반적으로 광범위한 목적에 적용됩니다. 이를 보여주기 위해 이전 실험에서 정의한 것과는 대조적으로 목표 #2 및 #3이라고 하는 두 가지 조정 목표를 정의합니다. 특히, 목적 #2는 식을 기반으로 유사하게 정의된다. (7), CPU 가격을 무시한다는 점을 제외하고(CPU 가격 = 0). 운영자는 주로 고비용 GPU를 더 잘 활용하는 데 관심을 갖기 때문에 이 정의는 프로덕션 클러스터에서 정당화됩니다. 목표 #3은 금전적 비용에 관계없이 최고의 RPS를 추구하도록 설정되어 있습니다. 그림 7은 Morphling의 탐색 비용과 두 가지 목표에 따른 5가지 기준선을 비교합니다. Morphling은 기준선에 비해 이점을 유지하며 항상 30개 샘플링 내에서 최적의 구성을 반환합니다. 그림 6b와 비교하여 BO는 두 가지 새로운 목표에서 급격한 효율성 저하를 확인했으며 최적의 구성을 찾기 위해 400개 이상의 샘플링(검색 공간의 55%)의 중앙값이 필요합니다. 우리는 가장 높은 RPS를 찾는 것과 같은 목표로 인해 종종 BO가 쉽게 고착될 수 있는 여러 로컬 최적이 있는 매우 고르지 않은 구성-성능 평면이 생성된다는 점에 주목합니다.

### 6.1.3 Microbenchmark

새로운 회귀 작업에 대한 빠른 적응. Morphling의 높은 효율성은 메타 모델을 새로운 추론 서비스에 빠르게 적응시키는 능력에 기인합니다. 이를 설명하기 위해 언어 모델 NNLM-128 [54]에 대해 GPU 메모리와 최대 배치 크기의 두 가지 구성 노브를 조정하는 것을 고려하고 그림 8에 메타 모델의 적응을 묘사합니다. 그림 8d는 구성 - 모델에 대해 수동으로 측정된 RPS 평면. 구성 샘플링 프로세스 동안 회귀의 목표는 이 매핑 평면을 맞추는 것입니다. 무화과 도 8a, 8b 및 8c는 초기 메타 모델 𝜃 *(초기 회귀 모델), 8개의 초기 샘플링 후 및 28개 샘플링 후의 적응 모델에 의해 제공된 매핑 평면을 각각 시각화합니다. 메타 모델은 초기 단계에서 8개의 고정 구성을 샘플링한 후 기본 정보에 빠르게 적응합니다. 28번의 샘플링 직후, 미세 조정된 회귀는 대상에 정확하게 맞출 수 있습니다. 이것은 Morphling이 몇 번의 샷으로 전체 최적값을 찾을 수 있는 이유를 설명합니다. 이에 비해, Fig. 9는 28번의 샘플링 후에 장착된 평면이 ground truth와 멀리 떨어져 있는 동일한 모델 NNLM-128[54]에 대한 BO의 피팅 과정을 시각화합니다.

검색 경로. 빠른 적응과 함께 Morphling의 검색 경로를 추가로 설명하기 위해 고정된 초기 샘플링 후 처음 10개의 샘플링된 구성을 설명합니다. Universal-Sentence-Encoder[20](notion://www.notion.so/un.se.en)와 EfficientNetb5[59](notion://www.notion.so/effic.5)의 두 가지 ML 모델을 고려합니다. 조정 목표는 Eq.로 설정됩니다. (7). 철저한 프로파일링은 effic.5 및 un.se.en에 대한 최적의 구성이 ⟨3 CPU 코어, 5% GPU 메모리, V100, 배치 크기 8⟩ 및 ⟨1 CPU 코어, 10% GPU 메모리, T4, 배치 크기 128임을 보여줍니다. ⟩, 각각.
그림 10a는 2차원 공간에서 두 모델의 탐색 경로를 시각화한 것이다. 두 검색 모두 동일한 지점(1 CPU 코어, 배치 크기 16)에서 시작하지만 각각의 최적(별표로 표시)으로 이어지는 다른 경로로 확장됩니다. 유사한 결과가 그림 10b에도 나와 있습니다. 여기서 Morphling은 T4가 un.se.en에 가장 적합하고 V100이 effic.5에 가장 적합하다는 것을 빠르게 식별합니다.

## 참조

- 웹사이트
    
    [https://kubedl.io/docs/prologue/introduction/](https://kubedl.io/docs/prologue/introduction/)
    
- 깃헙
    
    [https://github.com/kubedl-io/morphling](https://github.com/kubedl-io/morphling)
    

### Model-agnostic meta-learning for fast adaptation of deep networks.

- 논문
    
    [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3ea58384-4ca1-43fd-b812-07eb4b210faf/Model-Agnostic_Meta-Learning_for_Fast_Adaptation_of_Deep_Networks.pdf)
    
- 고려대 랩실
[http://dmqm.korea.ac.kr/activity/seminar/265](http://dmqm.korea.ac.kr/activity/seminar/265)
    
    세미나 참고 자료
    
    [0719세미나_목충협.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/67ab5712-2520-4b88-b0b6-b7129a969124/0719세미나_목충협.pdf)
    

### A tutorial on Bayesian optimization.

- 논문
    
    [A Tutorial on Bayesian Optimization.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e6135ad6-7627-47bc-9912-f9a8ffe54429/A_Tutorial_on_Bayesian_Optimization.pdf)
