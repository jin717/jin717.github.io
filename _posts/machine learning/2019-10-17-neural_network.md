---
layout: post
title: 인공신경망(1)
category: machine_learning
tags: [neural network, 인공신경망, 퍼셉트론]
comments: true
---


## Neural Network의 정의
- 인간의 뇌를 구성하는 기본 단위인 뉴런(neuron)을 모방한 네트워크

## 뉴런의 구조
![neuron](/public/machine_learning/neuron.png)
- 수상 돌기(Dendrites)에서 외부 신호를 수용하고 축색 돌기(Axon)을 통해 신호를 출력한다.

## 뉴런 구조 모델링
![neuron_model](/public/machine_learning/neural_model.png)
- 뉴런에 있는 각 축색돌기들은 시냅스(synapse)라고 하는 접점을 통해 외부 뉴런과 연결된다.
- 뉴런은 입력으로 들어오는 여러 개의 신호들을 하나로 합산한 다음 활성함수(Activation Function)을 통해 다시 다른 뉴런의 입력으로 들어가게 된다.
- 위와 같은 뉴런을 여러 개로 구성하여 네트워크를 구성할 수 있으며 ANN(Artificial Neural Network)이라 한다.

## 퍼셉트론
<center><img src="/public/machine_learning/perceptron.png"></center>

- ANN의 가장 기본 구조 중 하나로 입력에 각각 가중치를 곱하고 합치고 계단함수(활성함수)를 적용하여 그 결과(y)를 출력합니다.
- $$ \quad y\quad =\quad \left\{ \begin{matrix} 0,\quad ({ w }_{ 1 }{ x }_{ 1 }\quad +\quad { w }_{ 2 }{ x }_{ 2 })\quad <\quad \theta  \\ 1,\quad ({ w }_{ 1 }{ x }_{ 1 }\quad +\quad { w }_{ 2 }{ x }_{ 2 })\quad \ge \quad \theta  \end{matrix} \right  (단, \theta 는 계단함수의 임계치) $$

## 퍼셈트론으로 논리회로 학습
- AND 게이트의 진리표

| $$ { x }_{ 1 } $$ | $$ { x }_{ 2 }$$ | $$ y $$ |
|:--------:|:--------:|:--------:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | 1 |

- 위 진리표를 만족하는 모델(퍼셉트론)의 매개 변수인 (𝑤_1,𝑤_2, 𝜃)의 조합은 (0.5, 0.5, 0.7), (0.5, 0.5, 0.8), (1.0, 1.0, 1.0)등 무수히 많다.
- NAND  게이트와 OR 게이트 또한 특정 매개 변수가 주어지면 표현 가능하다.
- XOR 게이트는 표현 불가능하다. (단일 퍼셉트론은 비선형 영역을 표현할 수 없다.)
- 실제 데이터 분석(머신러닝) 과정 중에서는 학습 단계에서 적절한 매개 변수를 찾아간다.


참조 : 밑바닥부터 시작하는 딥러닝 - 사이토 코키
