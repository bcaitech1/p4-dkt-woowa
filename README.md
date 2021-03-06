# μ°μνκ², DKT

<p align="center">

![Image from iOS](https://user-images.githubusercontent.com/37537248/122682759-4c978f00-d236-11eb-84b6-2c5f37ee9413.gif)

</p>

<p align="center">
    <img src="https://img.shields.io/badge/python-v3.7-blue?logo=python" />
    <img src="https://img.shields.io/badge/pytorch-v1.9-blue?logo=pytorch" />
    <img src="https://img.shields.io/badge/pandas-v1.2.4-blue?logo=pandas" />
  </p>
  <p>π 2021.05.24 ~ 2021.06.16</p>
  <span style="font-weight:600">μλνμΈμ, DKTλ§μ  μ°μνκ². μ°μν μΊ νΌλ€μλλ€.</span>
</p>

> μ΄μ PR μΈμ λ νμμλλ€. π

## β¨ LB Score

### Public Leaderboard
***Accuracy: 0.7527 (10th), AUROC: 0.8226 (11th)***

### Private Leaderboard
***Accuracy: 0.7608 (2nd), AUROC: 0.8322 (9th)***

## π GOAL (νλ‘μ νΈ λͺ©ν)
```
Iscream νλ‘κ·Έλ¨μ μ΄μ©νλ νμλ€μ λ¬Έμ  νμ΄ μ΄λ ₯ μνμ€λ₯Ό ν΅ν΄ κ° νμμ΄ λ§μ§λ§ λ¬Έν­μ λ§μΆμ§ νλ¦΄μ§ μμΈ‘
```

## π§Ύ Introducton

### π¨βπ« Deep Knowledge Tracing
  - DKTλ Deep Knowledge Tracingμ μ½μλ‘ μ°λ¦¬μ "μ§μ μν"λ₯Ό μΆμ νλ λ₯λ¬λ λ°©λ²λ‘ μλλ€.


### π§βπ Problem
![image](https://user-images.githubusercontent.com/82928126/122641582-3e1b7b80-d141-11eb-91c3-3a06e27ac0fe.png)
- κ° νμμ΄ νΌ λ¬Έμ  λ¦¬μ€νΈμ μ λ΅ μ¬λΆκ° λ΄κΈ΄ λ°μ΄ν°λ₯Ό λ°μ μ΅μ’ λ¬Έμ λ₯Ό λ§μΆμ§ νλ¦΄μ§ μμΈ‘νμμ΅λλ€.
- μ΄λ² λνμμλ Iscream λ°μ΄ν°μμ μ΄μ©νμ¬ DKT λͺ¨λΈμ κ΅¬μΆνμκ³ , νμ κ°κ°μΈμ μ΄ν΄λλ₯Ό κ°λ¦¬ν€λ μ§μ μνλ₯Ό μμΈ‘νλ μΌλ³΄λ€λ μ£Όμ΄μ§ λ¬Έμ λ₯Ό λ§μΆμ§ νλ¦΄μ§ μμΈ‘νλ κ²μ μ§μ€νμμ΅λλ€.
- μ°λ¦¬λ κ° νμμ΄ νΌ λ¬Έμ  λ¦¬μ€νΈμ μ λ΅ μ¬λΆκ° λ΄κΈ΄ λ°μ΄ν°λ₯Ό λ°μ ***κ° μ¬μ©μ λ³λ‘ μ΅μ’ λ¬Έμ λ₯Ό λ§μΆμ§ νλ¦΄μ§ μμΈ‘νμμ΅λλ€.***

### π Dataset
- `.csv` ννλ‘ μ κ³΅λλ 7442 λͺμ Iscream μ¬μ©μμ λ¬Έμ νμ΄ λ°μ΄ν° 
- `train` : `test` = 9 : 1 (κ°κ° 6698, 744 λͺ)
-  `νμ΅ λ°μ΄ν°`</br>
    <img width="857" alt="dataset" src="https://user-images.githubusercontent.com/37537248/122682666-d561fb00-d235-11eb-91c5-a46ee22d3520.png"> </br>
    - `userID`: μ¬μ©μμ κ³ μ λ²νΈ
    - `assessmentID`: λ¬Έν­μ κ³ μ λ²νΈ (μ΄ 9454 κ°)
    - `testId`: μνμ§μ κ³ μ λ²νΈ (μ΄ 1537 κ°)
    - `answerCode`: μ¬μ©μκ° ν΄λΉ λ¬Έμ λ₯Ό λ§μ·λμ§ μ¬λΆ - `0`: μ€λ΅, `1`: μ λ΅)
    - `Timestamp`: μ¬μ©μκ° ν΄λΉλ¬Έν­μ νκΈ° μμν μμ μ λ°μ΄ν°
    - `KnowledgeTag`: λ¬Έν­μ μ€λΆλ₯ νκ·Έ (μ΄ 912 κ°)
-  `νκ° λ°μ΄ν°`
   -  νμ΅ λ°μ΄ν°μ κ°μ κ΅¬μ‘°
   -  `Timestamp` μ κ°μ₯ λ§μ§λ§ λ¬Έν­μ `answerCode`λ λͺ¨λ `-1`λ‘ νμλμ΄ μκ³ , ν΄λΉ λ¬Έμ μ μ λ΅ μ¬λΆλ₯Ό λ§μΆλ κ²μ΄ κ³Όμ 

### π λλ ν λ¦¬ κ΅¬μ‘°

<details>
<summary>Baseline</summary>
  <div markdown="1">
    
```
π Project Folder  
πsequential-model
βββ train
βββ inference
βββ args
βββ πdkt
    βββ creterion
    βββ custom_model
    βββ dataloader
    βββ features
    βββ metric
    βββ model
    βββ modeloptimizer
    βββ scheduler
    βββ temp
    βββ tranier
    βββ utils
```
</details>

## π‘ λ¬Έμ  ν΄κ²° μ λ΅

### β LGBM (Light Gradient Boosting Machine) λͺ¨λΈμ μ£Όλ ₯μΌλ‘ νκ² λ κ³κΈ°
```markdown
1. λ₯λ¬λ λͺ¨λΈμ΄ μ₯μ μ μ λλ‘ λ°ννκΈ°μλ λ€μ λΆμ‘±νλ λ°μ΄ν°μ
    - Riiid λ°μ΄ν°μ λΉν΄ νμ΅ λ°μ΄ν°κ° λ§μ΄ λΆμ‘±νμ (7442 κ°)
2. LGBMμ μ νλ°μ΄ν°μ μμ£Ό νμ©λλ λνμ μΈ Gradient Boosting λͺ¨λΈ Catboostμ XGBoostμμ λνλλ overfittingκ³Ό μλ λ¬Έμ λ₯Ό κ°μ ν λͺ¨λΈ
    - Catboostμ λ¬Έμ μ : λ¨μ μ ν λ°μ΄ν°μ λν overfitting κ°λ₯μ±
    - XGBoostμ λ¬Έμ μ : level-wise νΈλ¦¬ λͺ¨λΈ -> λͺ¨λ  λΈλμμ depthλ₯Ό λλ €κ°κΈ° λλ¬Έμ λΉν¨μ¨μ 
3. Feature engineeringμΌλ‘ custom featureλ€μ μΆκ°νμ λ λ€λ₯Έ λ₯λ¬λ λͺ¨λΈλ€μ λΉν΄ LGBMμ μ±λ₯μ΄ κ°μ₯ ν₯μ λμμ
4. Feature engineering κ²°κ³Ό sequential featureκ° μλ λ€λ₯Έ featureλ€μ feature importanceκ° λκ² μΈ‘μ λμμ
```
---

### 1οΈβ£ Single Model μ±λ₯ λΉκ΅
*`LGBM`κ³Ό `LSTM-Attention` λͺ¨λΈμμ λ¦¬λλ³΄λ μ§νκ° κ°μ₯ λμμ* </br>
<img width="633" alt="single_model_scores" src="https://user-images.githubusercontent.com/37537248/122682731-2114a480-d236-11eb-893c-92d5dfb873c4.png">

### 2οΈβ£ Feature Engineering & Feature Selection
```markdown
1. Feature Engineering
    1. User κ΄λ ¨ features
        - μ λ΅λ₯ 
        - λ¬Έμ  νμ΄ μκ°
        - λμ΄λ
    2. Question κ΄λ ¨ features
        - μ λ΅λ₯ 
        - λμ΄λ
2. Feature Selection
    - LGBM λͺ¨λΈ νμ΅ κ²°κ³Όλ₯Ό λ°νμΌλ‘ LB AUROC ν₯μμ κΈ°μ¬νλ featureλ€ μ μ 
    - Feature Importance plot & μ¬κ· νΉμ± μ κ±°λ² (Recursive Feature Elimination)μ νμ©νμ¬ μ μ²΄ custom feature μ€ λͺ¨λΈ νμ΅μ νμ©ν  featureλ€ μ μ 
```
- Custom Features</br>
  ![feature_list](https://user-images.githubusercontent.com/37537248/122682741-2e319380-d236-11eb-8efc-408df1553c04.gif)
- Feature Importance Plot</br>
![feature_importance](https://user-images.githubusercontent.com/37537248/122682704-03dfd600-d236-11eb-917f-a989bf77c17b.png)

### 3οΈβ£ Data Augmentation
```markdown
1. Before
    - κ° μ¬μ©μ λ³λ‘ κ°μ₯ μ΅κ·Όμ νΌ n κ°μ λ¬Έν­ λ°μ΄ν°λ§μ μνμ€λ‘ μ΄μ© (n: μ΅λ μνμ€ κΈΈμ΄)
    - λ¬Έμ  νμ΄ μ΄λ ₯μ΄ λͺ¨λΈμ μ΅λ μνμ€ κΈΈμ΄λ³΄λ€ κΈ΄ κ²½μ° λ°μ΄ν°λ₯Ό νμ©νμ§ λͺ»νκ³  λ²λ¦¬κ² λ¨
2. After
    - 'sliding window' λ°©μμ νμ©νμ¬ μ¬μ©μ λ³λ‘ μ΅λ μνμ€ κΈΈμ΄λ§νΌμ λ°μ΄ν°λ₯Ό μ¬λ¬ κ° μμ±
       -> λ¬Έμ  νμ΄ μ΄λ ₯μ΄ κΈΈλλΌλ μ΅λν νμ΅ λ°μ΄ν°λ‘ νμ©
    - 'sliding window': window sizeμ strideλ₯Ό μ§μ ν΄μ augmentation μ λλ₯Ό κ²°μ  κ°λ₯
```
<img width="779" alt="augmentation" src="https://user-images.githubusercontent.com/37537248/122682640-b19eb500-d235-11eb-8712-7324d2bedbe2.png">

### 4οΈβ£ CV Strategies
```markdown
* λ€μν K-foldμ λν μλ λ° κ²μ¦
    - Userλ³ Fold Split
    - Time series κΈ°μ€μΌλ‘ Fold Split
    - Label λΉμ¨ μ μ§νλ©°, Random shuffle
* κ²°λ‘ μ μΌλ‘λ μ μ²΄ randomν 30%μ userλ§μ validation setμΌλ‘ κ°μ Έμ¨ κ²½μ° κ°μ₯ λμκΈ°μ ν΄λΉ λ°©λ²μ μ ν
```
![image](https://user-images.githubusercontent.com/37537248/122683540-ca5d9980-d23a-11eb-9c81-fa853bfcf146.png)

## π©βπ©βπ§βπ¦Members
  |<img src="https://avatars.githubusercontent.com/u/42639690?v=4" width=300/>|<img src="https://avatars.githubusercontent.com/u/69613571?v=4" width=300/>|<img src="https://avatars.githubusercontent.com/u/37537248?v=4" width=300/>|<img src="https://avatars.githubusercontent.com/u/76548813?v=4" width=300/>|
  |:-:|:-:|:-:|:-:|
  |κΉνκ²°|λ₯μ§μ|μ±μΈκ·|μ΄λ€ν|
  | [@1gyeol-KIM](https://github.com/1gyeol-KIM) | [@jiwon-ryu](https://github.com/jiwon-ryu) | [@staycozyboy](https://github.com/staycozyboy) | [@dhh0](https://github.com/dhh0) |

## π Show your support
λ€λ€ λ€νΈμνΉ λ°μ΄ λ λ΄¬μπ
