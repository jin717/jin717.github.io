---
layout: post
title: Deep Neural Network(3)
category: machine_learning
tags: [neural network, loss function, gradient decent, gradient vanishing/exploding]
comments: true
---

## DNN의 문제점

지금의 DNN은 상당히 좋은 성능을 보이지만 초기에는 몇몇 문제점이 있었다.<br>
DNN 구조를 구성할 때 layer를 많이 쌓으면서 다음과 같다.

- Vanishing gradient
- Exploding gradient
- 학습 속도가 느리다.
- 많은 매개변수를 가지게 되면서 Overfitting을 하게 된다.

이런 문제점을 해결하기 위해 다음과 같은 다양한 방법에 대해 소개하려고 한다.

- 가중치 초기화
- 활성 함수 변경
- 배치 정규화
- 최적화 알고리즘 변경

## Vanishing & Exploding Gradient problem

- DNN의 학습 단계에서 경사 하강법으로 back-propagation을 통해 하위층으로 진행될 때, 아래와 같은 문제로 학습이 제대로 되지않는다.
    - 그레디언트가 점점 작아지면 가중치가 변경되지 않아 학습이 진행되지 않는 경우가 많이 발생한다.(Vanishing Gradient)
    - 반대로 그레디언트가 점점 커져 여러 개의 층이 비정상적으로 큰 가중치로 갱신되어 발산하게 된다.(Exploding Gradient)
    
- 세이비어 글로럿과 요슈아 벤지오의 논문 “Understanding the Difficulty of Training Deep Feedforward Neural Networks”(2010)에서 다음과 같은 문제점을 제시한다.  

### 문제 원인

- 각 layer를 통과하면서 입력값의 분산보다 출력값이 분산이 커짐
- 활성 함수로 시그모이드 함수를 사용하면서 입력의 분산이 커질 수록 0,1에 가까워 짐
- 평균이 0.5라는 점과 함수가 항상 양수를 출력하기 때문에 문제 발생(편향 이동)
<center><img src="/public/machine_learning/sigmiod_gradient.png" height="300"></center>

### 해결 방안
- 가중치 초기화 방법 변경
    - 세이비어 초기화
        - 평균이 0이고 표준편차가 $$ \sigma \quad =\quad \sqrt { \frac { 2 }{ { n }_{ input }+{ n }_{ output } }  } $$ 인 정규분포에서 무작위로 초기화
    - he 초기와
    - 그 외 활성 함수에 따라 다양한 초기화 방법이 있다.

- 활성 함수 변경
    - 하이퍼볼릭 탄젠트 : $$ \tanh { (x)\quad =\quad \frac { 2 }{ 1+{ e }^{ -2x } } -1 } $$
        - 장점 : 편향 이동이 발생하지 않는다.
        - 단점 : 시그모이드와 마찬가지로 𝑥 의 절대값이 커지면 그레디언트가 0에 수렴.
        <center><img src="/public/machine_learning/tanh_function.png" height="200"></center>
    - ReLU(rectified linear units) : $$ ReLU(x)\quad =\quad max(0,x) $$
        - 장점 : 계산이 간단하며 그레디언트가 유지된다.
        - 단점 : 𝑥 <0일 때, 그레디언트가 0이 되어 가중치를 변화하지 못하고 0만 출력하게 된다. (dying $$ ReLU $$)
        <center><img src="/public/machine_learning/ReLU_function.png" height="300"></center>
    - LeakyReLU : $$ LeakyReLU(𝑥)=max(0.01x,x) $$
        - 장점 : dying $$ ReLU $$ 해결($$ x < 0 $$ 일 때, 그레디언트가 0이 아님.)
        - 단점 : 0에서 미분 불가
	    - 참고) 기타 $$ LeakyReLU $4의 변종도 있음, $$ RReLU $$(randomized leaky ReLU)/$$ PReLU $$(parametric leaky ReLU)
	    <center><img src="/public/machine_learning/LeakyReLU_function.png" height="300"></center>
    - ELU : $$ { ELU }_{ \alpha  }\begin{cases} x,\quad x\quad >\quad 0 \\ \alpha ({ e }^{ x }\quad -\quad 1),\quad x\quad \le \quad 0 \end{cases} $$
        - 장점 : $$ 𝑥=0 $$일 때, 그레디언트가 급격하게 변하지 않음 (미분은 불가능)
        - 단점 : 계산속도가 상대적으로 느림
        <center><img src="/public/machine_learning/ELU_function.png" height="300"></center>
- \[ $$ ELU > LeakyReLU > ReLU > tanh > sigmoid $$] 순으로 사용하는 것을 권장한다.

### 배치 정규화


### 그레이언트 클리핑


## 학습 속도 향상

### Gradient descent 업그레이드

## Overfitting 방지

### early stopping
### dropout
### $$ { l }_{ 1 } $$과 $$ { l }_{ 2 } $$ regularization
### max-norm regularization


## 참고 자료
- Hands-on machine learning with Scikit-learn & Tensorflow - 오렐리앙 제롱
- 가중치 초기화 - https://reniew.github.io/13/
- 활성 함수 - https://ratsgo.github.io/deep%20learning/2017/04/22/NNtricks/
- 배치 정규화 - http://sanghyukchun.github.io/88/




