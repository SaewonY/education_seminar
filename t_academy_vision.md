# - 2019.05.16 tacademy vision 교육 

<br>

## 1. 딥러닝 일반 설명

<br>

+ DNN은 **함수 근사화 능력**이 있다.
+ 입출력 쌍을 제공하여 DNN 내부의 weight 값을 업데이트 한다.
+ 이를 위해 내부적으로 BP와 GD 알고리즘을 사용한다.
+ 충분한 컴퓨팅 자원이 필요하다.
+ 이를 반복하여 함수를 근사화하는 것이 딥러닝.

<br>

+ 전문가의 지식을 하드코딩 할 수도 있고 (전문가 시스템)
+ 데이터에서 로직을 찾을 수 있다. (머신러닝)

<br>

+ 최대 단점은 **비싼 비용** (데이터 수집) == 레이블링 데이터를 구하기 힘들다


<br>

## 2. 딥러닝 기술 용어

#### 1) **cost function**

+ MSE (Mean Squared Error)
+ CE (Cross Entropy)
+ KL-Divergence 
+ MLE (Maximum Likelihood Estimation)

<br> 

#### 2) Optimizer 
+ GD (Gradient Descent)
+ Batch GD
+ Mini-batch GD
+ SGD (Stochastic GD)
+ Momentum
+ Adagrad
+ Adadelta
+ Adam
+ RMsprop

<br>

#### 2) Optimizer
+ Dropout
+ BN (Batch Normalization)
+ Regularization
+ Data augmentation

<br>

#### 3) Activation Function (활성화 함수)
+ sigmoid (=logistic)
+ Tanh (between -1, 1)
+ Relu 
+ Leaky Relu

<br>

#### 4) Learning Rate
+ 가중치가 변화되는 정도

<br>

#### 5) Softmax
+ 최종 출력층에 사용되며 여러 개의 출력 노드의 합이 1이 되도록 만든다 -> 확률로 해석할 수 있게 된다.

#### 6) Gradient Descent
+ 가중치를 업데이트하기 위한 알고리즘. 비용함수의 값이 최소로 가도록 만든다.
  
<br>

#### 7) Backpropagation
+ 출력된 값과 원하는 값과의 차이를 가지고 그 전의 weight 값들을 변경하는 알고리즘
+ 뒤에서부터 오차값이 전파되기 때문에 back이라는 이름이 붙었다.
+ 실제 변경되는 값은 GD로 결정된다.

<br>

## 3. CNN (Convolutional Neural Network)
+ data로부터 feature extraction하는 방법
+ extract 한 이후에는 dense layer 등을 쌓으면서 classification 문제를 푼다.
+ 자세한 사항은 생략

<br>
<br>



## 4. 영상

#### 1) 영상 데이터

+ 영상은 픽셀로 구성된다.
+ 회색의 경우 1개의 픽셀은 밝기의 값 1개로 구성된다.
+ 칼라의 경우 RGB 3개의 채널, RED, GREEN, BLUE로 구성된다.

<br>

+ bitmap은 pixel 값 그대로 저장
+ gif, jpeg, png 등은 압축하여 저장한 것.

<br>


#### 2) 영상 분류 (Classification)

+ 대상을 미리 정해진 클래스(class)로 분류하는 작업
+ 영상의 경우 해당 영상이 무엇인지 인식하는 작업

<br>

#### 3) Imagenet (이미지넷)

+ 일반적인 사물 영상의 데이터셋
+ 1000개 클래스

<br>

#### 4) 영상 분류 현황
+ 완숙된 상태
+ keras를 설치하면 vgg나 resnet이 default로 포함되어 있다.
+ 그냥 가져다 쓰면 된다.

<br>

## 5. Object Detection

+ **IOU** (Intersection over union) - 정답과 탐지해낸 결과의 영역 일치 정도 (최대 1, 최소 0)
+ **MAP** (mean average precision) - 여러개의 물체가 탐지된 경우 각 정확도의 평균
+ **Average Recall** - 탐지해야 할 물체 중 정확히 탐지한 갯수의 비율의 평균

<br>

#### 1) YOLO (You Only Look Once)
+ 영역의 박스와 해당 박스의 분류를 동시에 복수개 출력하는 네트워크. 모두 98개 (7*7*2)가 제안되며 중복되지 않고 확신도가 높은 것만 추린다.
+ 장점 - speed, less mistake with background (전체 이미지를 학습함으로), highly generalizable (general한 이미지로 학습해도 잘된다)
+ 단점 - 작은 물체 detect가 잘 안된다.

<br>

![yolo](https://user-images.githubusercontent.com/40786348/57838237-4769e980-77ff-11e9-9f7b-3217bf1f4141.PNG)

<br>

#### 2) SSD (single shot detector)
+ vgg16을 기반으로 한다.
+ fully conntected layer를 없애고 각 conv 레이어의 값을 하나로 받는다.
+ yolo가 98개를 출력하는 것과 달리, ssd는 5776개의 박스를 출력한다. 

<br>

![image](https://user-images.githubusercontent.com/40786348/57838190-31f4bf80-77ff-11e9-8b3d-546a02993bd8.png)

<br>
<br>

### reference : Tacademy 딥러닝 영상분류 / 영상인식 입문
https://tacademy.skplanet.com/front/tacademy/courseinfo/campus.action?classIndex=1592