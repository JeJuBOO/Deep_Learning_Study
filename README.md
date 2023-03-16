# Deep Learning  
An MIT Press book.    
Ian Goodfellow and Yoshua Bengio and Aaron Courville.  
2016.  
https://www.deeplearningbook.org/ 

---
*deep Learning을 공부하기 위한 이 책을 쉽게 이해할 수 있게 정리 할 것이다.*

---
* Table of Contents
* [Acknowledgements](https://www.deeplearningbook.org/contents/acknowledgements.html)
* Notation
* 1.Introduction
* Part I Applied Math and Machine Learning Basics
  - 2.Linear Algebra
  - 3.Probability and Information Theory
  - 4.Numerical Computation
  - 5.Machine Learning Basics
* Part II: Modern Practical Deep Networks
  - 6.Deep Feedforward Networks
  - 7.Regularization for Deep Learning
  - 8.Optimization for Training Deep Models
  - 9.Convolutional Networks
  - 10.Sequence Modeling: Recurrent and Recursive Nets
  - 11.Practical Methodology
  - 12.Applications
* Part III: Deep Learning Research
  - 13.Linear Factor Models
  - 14.Autoencoders
  - 15.Representation Learning
  - 16.Structured Probabilistic Models for Deep Learning
  - 17.Monte Carlo Methods
  - 18.Confronting the Partition Function
  - 19.Approximate Inference
  - 20.Deep Generative Models
* Bibliography
* Index

# Capter 1. Introduction
 
> The true challenge to artiﬁcial intelligence proved to be solvingthe tasks that are easy for     people to perform but hard for people to describe formally—problems that we solve intuitively,   that feel automatic, like recognizingspoken words or faces in images.
> 인공지능에 대한 진정한 도전은 사람들이 수행하기는 쉽지만 공식적으로 설명하기 어려운 작업, 즉 우리   가 직관적으로 해결하고, 이미지에서 말하는 단어나 얼굴을 인식하는 것처럼 자동적으로 느껴지는 문제   를 해결하는 것임이 입증되었다.

![image](https://user-images.githubusercontent.com/71332005/225519582-e06acf2c-3e1b-4a25-9fd2-d27dfecece88.png)  

우리는 위 사진에서 머핀과 치와와를 구분하는것은 쉽다. 이처럼 그냥 우리가 자연스럽게 행하는 결정들 말하고 있다. 이를 컴퓨터가 해결하기 위해서 컴퓨터가 우리과 같이 경험으로부터 배우고 더 단순한 개념과의 관계를 통해 이해하도록 한다. 따라서 우리는 컴퓨터가 필요하는 지식을 임의로 알려주지 않음으로 컴퓨터는 더 단순한 개념으로 복잡한 개념을 학습할 수 있다.

1970년대의 로봇 공학자인 한스 모라벡(Hans Moravec)은 "Hard problems are easy and easy problems are hard.(어려운 일은 쉽고, 쉬운 일은 어렵다.)"라는 말을 했다. 내가 어떻게 자연스럽게 걷고 있는가를 공학적으로 생각해 보면 어떠한 프로세스가 있기야 하겠지만 너무나도 자연스러운 행동이기에 설명하기가 쉽지 않다. 로봇들이 이족 보행을 하기 위해서는 매우 복잡한 계산과정과 분석이 필요하기에 이와같은 말들이 나온것 같다. 하지만 로봇은 현재 산을 걸어 올라가고 춤을 추며 매우 발전했다. 한스 모라벡의 역설은 꺠지고 있는것이다.

나는 가끔 무언가를 하는것에 의미를 두지않고 그냥... 하는 일들이 있다. 효율적이지 않고 아무 의미없는 행동들. 만약 미래의 인공지능이 내린 결정의 근거가 '그냥' 이라면 그때는 공상영화와 같은 일들이 일어날 것 같다는 생각을 하게됬다.

### 1.1 Who Should Read This Book?
 딥러닝및 인공지능 연구분야에서 경력을 시작하는 대학생과 제품이나 플랫폼에서 딥러닝을 사용하기를 원해 빠르게 딥러닝을 습득하려하는 소프트웨어 엔지니어이다.  
 이 책은 독자들이 컴퓨터 공학을 배경으로하는 출신이라고 가정한다. 프로그래밍, 계산 성능 문제에 대한 기본적인 이해, 복잡성 이론, 입문 수준 미적분학 및 그래프 이론의 일부 용어에 익숙하다고 가정한다.

### 1.2 Historical Trends in Deep Learning
어떤 역사적 맥락에서 딥러닝을 이해하는 것은 가장 쉽다. 딥 러닝에 대한 자세한 이력을 제공하기보다는 몇 가지 주요 동향을 식별한다.
- 딥 러닝은 길고 풍부한 역사를 가지고 있지만, 다양한 철학적 관점을 반영하면서, 많은 이름으로 알려졌고 인기가 시들해졌다.
- 사용 가능한 훈련 데이터의 양이 증가함에 따라 딥 러닝이 더욱 유용해졌다.
- 딥 러닝 모델은 딥 러닝을 위한 컴퓨터 인프라(하드웨어와 소프트웨어 모두)가 개선됨에 따라 시간이 지남에 따라 규모가 커졌다.
- 딥 러닝은 시간이 지남에 따라 점점 더 복잡해지는 애플리케이션을 정확하게 해결했습니다

### 1.2.1 The Many Names and Changing Fortunes of Neural Net-works
  딥러닝이 최신 기술로 생각하는 사람들이 많을것이다. 하지만 딥러닝의 시작은 1940년으로 거슬러 올라간다. 1940년대부터 1960년대까지 사이버네틱스, 1980년대부터 1990년대까지 커넥션리즘, 2006년부터 딥러닝이라는 이름으로 부활한 세 가지 발전의 물결이 있다.  
 또한 생물학적 학습 계산 모델, 즉 생물학적 뇌에서 영감을 받은 이름 중 하나인 인공신경망(ANN)이다.  
 현대 용어 "딥 러닝"은 현재의 기계 학습 모델에 대한 신경과학적 관점을 넘어선다. 그것은 여러 수준의 구성을 학습하는 보다 일반적인 원칙에 호소하며, 이는 반드시 신경적으로 영감을 받지 않는 기계 학습 프레임워크에 적용될 수 있다.

1940년대 입력값과 출력값을 연관시키기 위해 가중치를 학습하는 간단한 선형 모델부터 시작했다. 1950년대 퍼셉트론은 주어진 입력에 따라 가중치를 학습할 수 있는 최초의 모델이다. 이와 비슷한시기에 적응 선형 요소(ADALINE)(Widrowand Hoﬀ, 1960)은 실수를 예측하기 위해 단순히 x 자체의 값을 반환했고 데이터로부터 여러 숫자를 예측하는 방법을 배웠다. ADALINE의 가중치를 적용하는 데 사용된 훈련 알고리즘은 확률적 경사 하강법이라고 불리우는 오늘날까지 아주 중요하게 쓰이는 알고리즘으로 발전했다.

이후에 1980년대 인공신경망은 포유류 시각 시스템의 구조에서 영감을 받은 모델 아키텍쳐를 도압했고 이 모델은 현대의 컨볼루션 네트워크(LeCun et al., 1998)의 기반이 되었다.

현대는 우리의 생각과 '뇌'를 모방하려는 인공신경망에서 시작된 딥러닝이지만, 딥러닝 연구자들은 신경과학에 관심이 없고, 컴퓨터 신경과학으로 별개의 연구분야로 나뉘었다. 그럼에도 우리 뇌가 어떻게 작동하려는지 이해하려는 노력은 딥러닝의 발전에 큰 영향을 끼쳤다.

1990년대 매무 빠른속도로 발전해 온다. 이미지와 언어 등 다양한 발전을 맞이했고, 책에 많은 연구들이 나왔있다. 이는 현재의 다양한 이미지처리 모델과 ChatGPT같은 언어 처리 모델이 등장하기 까지 끊임없이 발전했다.

### 1.2.2 Increasing Dataset Sizes
1950년대 부터 시작됬음에도 최근에서 빠른 발전이 되고 있음은 시간이 지남에 따라 증가하는 데이터의 양 덕분일 것이다. 따라서 현재의 더 많은 데이터를 다루고 연구하는 것 또한 매우 중요한 연구 분야이다.

### 1.2.3 Increasing Model Sizes
 신경망이 크게 성공하는 큰 이유중 또 다른 하나는 더 깊고 큰 모델을 실행할 수 있기 때문이다. 이는 내가 초등학교 컴퓨터실에 큰 컴퓨터와 플로피 디스크를 썼던 때 에서 몇배로 빠르고 좋은 컴퓨터들이 즐비한 현재까지의 발전이 있었다. 더 빠른 CPU환경 에다가 GPU의 출현은 딥러닝 역사에 가장 중요한 추세 중 하나이다. 
 
### 1.2.4 Increasing Accuracy, Complexity and Real-World Impact
 딥러닝의 인식과 예측의 정확도는 꾸준히 향상했다. 또한 다양한 곳에서 성공적으로 적용되었다. 
 이는 다양한 챌린지와 연구 개발을 통해 발전되었고 현재 또한 다양한 챌린지에서 작년 성적을 뛰어넘고 있다. 
 그리고 연구된 딥러닝은 다양한 분에서 활용되고 또한 다양한 분야에서 활용되는 연구가 딥러닝의 발전에 큰 도움이 되고있다.
 
*소개가 굉장히 길어 놀랐지만 읽다보면 빠져들어 읽게된다. 비록 어떤부분은 Ctrl C+V 기술을 사용하여 작성했지만 찬찬히 읽어내려 가며 잘 작성된 책이라고 느끼며 소개를 위와같이 정리해 보았다.*

#
 
[맨위로](#deep-learning)

















