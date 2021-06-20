# 우아하게, DKT

<p align="center">

![Image from iOS](https://user-images.githubusercontent.com/37537248/122682759-4c978f00-d236-11eb-84b6-2c5f37ee9413.gif)

📆 2021.05.24 ~ 2021.06.16
</p>

<p align="center">
    <img src="https://img.shields.io/badge/python-v3.7-blue?logo=python" />
    <img src="https://img.shields.io/badge/pytorch-v1.9-blue?logo=pytorch" />
    <img src="https://img.shields.io/badge/pandas-v1.2.4-blue?logo=pandas" />
  </p>
  <span style="font-weight:600">안녕하세요, DKT마저 우아하게. 우아한 캠퍼들입니다.</span>
</p>

> 이슈 PR 언제나 환영입니다. 🙌

## ✨ LB Score

### Public Leaderboard
***Accuracy: 0.7527 (10th), AUROC: 0.8226 (11th)***

### Private Leaderboard
***Accuracy: 0.7608 (2nd), AUROC: 0.8322 (9th)***

## 📌 GOAL (프로젝트 목표)
```
Iscream 프로그램을 이용하는 학생들의 문제 풀이 이력 시퀀스를 통해 각 학생이 마지막 문항을 맞출지 틀릴지 예측
```

## 🧾 Introducton

### 👨‍🏫 Deep Knowledge Tracing
  - DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론입니다.


### 🧑‍🎓 Problem
![image](https://user-images.githubusercontent.com/82928126/122641582-3e1b7b80-d141-11eb-91c3-3a06e27ac0fe.png)
- 각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아 최종 문제를 맞출지 틀릴지 예측하였습니다.
- 이번 대회에서는 Iscream 데이터셋을 이용하여 DKT 모델을 구축하였고, 학생 개개인의 이해도를 가리키는 지식 상태를 예측하는 일보다는 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중하였습니다.
- 우리는 각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아 ***각 사용자 별로 최종 문제를 맞출지 틀릴지 예측하였습니다.***

### 📊 Dataset
- `.csv` 형태로 제공되는 7442 명의 Iscream 사용자의 문제풀이 데이터 
- `train` : `test` = 9 : 1 (각각 6698, 744 명)
-  `학습 데이터`</br>
    <img width="857" alt="dataset" src="https://user-images.githubusercontent.com/37537248/122682666-d561fb00-d235-11eb-91c5-a46ee22d3520.png"> </br>
    - `userID`: 사용자의 고유번호
    - `assessmentID`: 문항의 고유번호 (총 9454 개)
    - `testId`: 시험지의 고유번호 (총 1537 개)
    - `answerCode`: 사용자가 해당 문제를 맞췄는지 여부 - `0`: 오답, `1`: 정답)
    - `Timestamp`: 사용자가 해당문항을 풀기 시작한 시점의 데이터
    - `KnowledgeTag`: 문항의 중분류 태그 (총 912 개)
-  `평가 데이터`
   -  학습 데이터와 같은 구조
   -  `Timestamp` 상 가장 마지막 문항의 `answerCode`는 모두 `-1`로 표시되어 있고, 해당 문제의 정답 여부를 맞추는 것이 과제

### 🗂 디렉토리 구조

<details>
<summary>Baseline</summary>
  <div markdown="1">
    
```
🗃 Project Folder  
📁sequential-model
├── train
├── inference
├── args
└── 📁dkt
    ├── creterion
    ├── custom_model
    ├── dataloader
    ├── features
    ├── metric
    ├── model
    ├── modeloptimizer
    ├── scheduler
    ├── temp
    ├── tranier
    └── utils
```
</details>

## 💡 문제 해결 전략

### ❓ LGBM (Light Gradient Boosting Machine) 모델을 주력으로 하게 된 계기
```markdown
1. 딥러닝 모델이 장점을 제대로 발휘하기에는 다소 부족했던 데이터셋
    - Riiid 데이터에 비해 학습 데이터가 많이 부족했음 (7442 개)
2. LGBM은 정형데이터에 자주 활용되는 대표적인 Gradient Boosting 모델 Catboost와 XGBoost에서 나타나는 overfitting과 속도 문제를 개선한 모델
    - Catboost의 문제점: 단순 정형 데이터에 대한 overfitting 가능성
    - XGBoost의 문제점: level-wise 트리 모델 -> 모든 노드에서 depth를 늘려가기 때문에 비효율적
3. Feature engineering으로 custom feature들을 추가했을 때 다른 딥러닝 모델들에 비해 LGBM의 성능이 가장 향상 되었음
4. Feature engineering 결과 sequential feature가 아닌 다른 feature들의 feature importance가 높게 측정되었음
```
---

### 1️⃣ Single Model 성능 비교
*`LGBM`과 `LSTM-Attention` 모델에서 리더보드 지표가 가장 높았음* </br>
<img width="633" alt="single_model_scores" src="https://user-images.githubusercontent.com/37537248/122682731-2114a480-d236-11eb-893c-92d5dfb873c4.png">

### 2️⃣ Feature Engineering & Feature Selection
```markdown
1. Feature Engineering
    1. User 관련 features
        - 정답률
        - 문제 풀이 시간
        - 난이도
    2. Question 관련 features
        - 정답률
        - 난이도
2. Feature Selection
    - LGBM 모델 학습 결과를 바탕으로 LB AUROC 향상에 기여하는 feature들 선정
    - Feature Importance plot & 재귀 특성 제거법 (Recursive Feature Elimination)을 활용하여 전체 custom feature 중 모델 학습에 활용할 feature들 선정
```
- Custom Features</br>
  ![feature_list](https://user-images.githubusercontent.com/37537248/122682741-2e319380-d236-11eb-8efc-408df1553c04.gif)
- Feature Importance Plot</br>
![feature_importance](https://user-images.githubusercontent.com/37537248/122682704-03dfd600-d236-11eb-917f-a989bf77c17b.png)

### 3️⃣ Data Augmentation
```markdown
1. Before
    - 각 사용자 별로 가장 최근에 푼 n 개의 문항 데이터만을 시퀀스로 이용 (n: 최대 시퀀스 길이)
    - 문제 풀이 이력이 모델의 최대 시퀀스 길이보다 긴 경우 데이터를 활용하지 못하고 버리게 됨
2. After
    - 'sliding window' 방식을 활용하여 사용자 별로 최대 시퀀스 길이만큼의 데이터를 여러 개 생성
       -> 문제 풀이 이력이 길더라도 최대한 학습 데이터로 활용
    - 'sliding window': window size와 stride를 지정해서 augmentation 정도를 결정 가능
```
<img width="779" alt="augmentation" src="https://user-images.githubusercontent.com/37537248/122682640-b19eb500-d235-11eb-8712-7324d2bedbe2.png">

### 4️⃣ CV Strategies
```markdown
* 다양한 K-fold에 대한 시도 및 검증
    - User별 Fold Split
    - Time series 기준으로 Fold Split
    - Label 비율 유지하며, Random shuffle
* 결론적으로는 전체 random한 30%의 user만을 validation set으로 가져온 경우 가장 높았기에 해당 방법을 선택
```
![image](https://user-images.githubusercontent.com/37537248/122683540-ca5d9980-d23a-11eb-9c81-fa853bfcf146.png)

## 👩‍👩‍👧‍👦Members
  |<img src="https://avatars.githubusercontent.com/u/42639690?v=4" width=300/>|<img src="https://avatars.githubusercontent.com/u/69613571?v=4" width=300/>|<img src="https://avatars.githubusercontent.com/u/37537248?v=4" width=300/>|<img src="https://avatars.githubusercontent.com/u/76548813?v=4" width=300/>|
  |:-:|:-:|:-:|:-:|
  |김한결|류지원|성인규|이다혜|
  | [@1gyeol-KIM](https://github.com/1gyeol-KIM) | [@jiwon-ryu](https://github.com/jiwon-ryu) | [@staycozyboy](https://github.com/staycozyboy) | [@dhh0](https://github.com/dhh0) |

## 🌟 Show your support
다들 네트워킹 데이에서 봬요💖
