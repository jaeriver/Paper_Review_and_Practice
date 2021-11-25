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
    
- Mapping function (regression tasks)

## Abstract

meta-learning 을 이용해서 많은 구성요소 중 최적화된 모델을 찾는다. Alibaba와 비교.

## Introduction

- hardware configuration
    
    CPU cores, GPU types, GPU memory, GPU share(if gpu sharing is supported)
    
- Alibaba
    
    low resource utilization ( 80%의 cpu, gpu 메모리 할당량을 안씀)
    
- Prevalent auto-configuration techniques
    
    Bayesian optimization, linear regression, transfer learning
    
    저차원 구성 공간에서는 클라우드 작업 부하를 잘 조정하지만 작업 공간이 늘어나면 최적점을 찾기위한 large sampling overhead가 증가하여 비효율적이다.
    
- 다양한 모델임에도 similiar tendency가 있음 따라서 few-shot learning problem을 공식화하고 MAML 기술을 사용하여 다양한 하드웨어와 런타임 구성 환경에서 성능이 어떻게 변하는지 포착하고자 함.

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

GPU Memory는 최소 요구 조건 같은 느김

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
