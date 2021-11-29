**[A.I.D.D Competition](https://github.com/DatathonInfo/AIDD2021)** Review  
_Reviewed by 황동준_

|대회|제출 모델|정확도|순위|
|------|-----|-----|-----|
|본선|MLP|82.22|15 / 20|
|예선|MLP|82.08|4 / 40|

**Summary**
- `MLP` 에 너무 붙잡혀 있었음.
- `Tabular Data` 그리고 specificity, sensitivity 를 고려해야 하는 문제에서는 ML Algorithm 을 이용하면서 Data Transform 에 신경쓰자.
- `Imbalanced Data` 였기 때문에 이를 어떻게 해결 할지가 가장 큰 문제였다고 생각함.

# DataSet
당뇨병 데이터 
- `Tabular DataSet`
- `Binary Classification` 필요
- 22 Columns

# Code Summary
- `main.py`: logistic regression
- `main_dt.py`: Decision Tree
- `main_lr.py`: 예, 본선에 제출한 top model
- `main_xbnet.py`: xbnet 을 적용
- `main_best.py`: AdaBoostClassifier 최적화
- `setup.py`: nsml docker 에 설치할 python library

## Model tried
팀 전체를 보았을 때는 여러 개를 시도했지만 제가 시도했던 것은 다음과 같습니다.
### 1. Neural Net
MLP with Perceptron Rule's  
```python
Linear -> BatchNorm -> Sigmoid(ELU) -> Dropout
```
- Layer를 23(input), 16, 20, 24, 20, 16, 1(output) 으로 쌓음.
- Layer를 23(input), 16, 16, 10, 4, 1(output) 으로도 쌓음.
- Layer를 23(input), 16, 1(output) 으로도 쌓음.  

모델을 돌려본 결과 accuracy 는 오랜 epoch (최소 1000이상)를 돌렸을 때 모두 동일함.  
실제로 다음과 같은 Rule을 적용했는 데 유용했던 것과 유용하지 않았던 것이 있음.  

(1) input 에서 bias column 하나를 추가하자 (효과 x, 편차가 큰 데이터가 아니였나봄)  
(2) (input + output) * (2/3) = 16 만큼의 perceptron 을 첫 layer에 넣고, 
(input + output) 이상의 perceptron 을 layer에 넣지 마라. (overfitting을 어느정도 방지)  
(3) 위에 적어놓았던 Layer 순서로 학습진행 (BatchNorm과 Dropout 중 하나만 빼도 정확도 낮음.)

**LR Scheduler and optimizer**  
- LR은 `CosineWarmUpStarter`를 이용해서 0.1 부터 낮아질 수 있도록 했지만, 큰 효과 없고 오히려 MultiStep이 안정성이 높았음. -> 이는 dimension 이 낮기 때문에 local optima에 빠질 확률이 적음.
- optimizer은 `Adam`을 이용했는데, 처음에는 `weight decay`를 추가하지 않았다가 overfitting 이 심하게 일어나는 것 같아서 AdamW로 바꾸면서 `regularization` 을 추가함. (근데 overfitting 이 아니였음. 그래서 효과 x)

**Data Transform**
- Oversampling 은 `SMOTE` 이용 (ROSE보다 성능 좋음) -> `borderline, SVMSMOTE, SMOTEN, NCSMOTE` 시도 했으나 다 고만고만함. SMOTEN 은 심지어 성능 떨어짐 (imbalance 에서 데이터 셋을 편향되게 만들었음.)
- Scaler, Selector는 이용하지 않음. (했더니 성능 감소 -> MLP가 알아서 처리해줌.)

#### 보충해야 할 점  

- SMOTE 가 중요했었던 것 같음. Oversampling 할 때 propotion을 설정할 수 있었으면 좋았을 텐데, [smote_variants](https://smote-variants.readthedocs.io/en/latest/oversamplers.html#msmote) 라는 library 가 제대로 작도하지 않아서 적용하지 못했음.
- [smote_variants](https://github.com/analyticalmindsltd/smote_variants) 를 나중에라도 꼭 시도해 볼 것 (다른 데이터에라도)
- nsml 에서는 correlation 을 시각적으로 확인할 수 없었는데 그렇기 때문에 어떤게 잘못 sampling 되었는지 찾기 힘들었음.
- MLP에서는 이 구조가 너무 low layer 만 아니면 Layer, LR, Optimizer를 어떻게 조정하든 같은 값이 나오게 됨. -> 일단 feature 자체가 몇 없었기 때문에, MLP처럼 일반화 시키는, 데이터의 exception 을 가리기 힘든 데이터에서는 효과가 덜함.
- 즉 MLP에서 overfitting 만 안나는 구조면 성능은 거의 다 비슷함.

### 2. Neural Net + ML Algorithm
`XBNet`이 Boosting Algorithm과 Neural Net구조를 합친 Network인데, nsml에서 이용하기가 어려웠음. (train 데이터를 test에서 같이 받아올 수 있었으면 쉽게 가능했을 듯)

### 3. ML Algorithm

써본 Algorithm은 다음과 같음.
- Tree Network (Decision Tree)
- Neighbor Network (KNeighbors Classifier)
- Linear Model (LinearRegression, LogisticRegression)
- SVM (SVR)
- Ensemble (StackingRegressor, AdaBoostClassifier, RandomForestClassifier)

여기 중에서 `AdaBoostClassifier`, n_estimators = 55 가 가장 성능이 좋았음.

#### Data Transform
**Scaler**  
- Normalize, MinMax, Standard 중, Normalize가 안정적인 성능을 보임 (모든 column에 일괄 적용)
- 보충 할 점에서 추가 설명하겠음.

**Oversampling**  
- ROSE가 가장 뛰어남. 이는 SMOTE같은 경우 좀 더 많은 데이터를 sampling 하고, 적은 데이터의 것을 늘리면서 noise 가 크게 발생하면서 `maximum exception data`가 발생하기 때문에 ML Algorithm에서 성능이 좋지 못함. ROSE는 minority와 majority를 적절히 조절하기 때문에 noise 가 덜함.
- 다른 SMOTE도 써봤지만 ROSE가 제일 좋음 (RandomOverSampler)

**Selector**
- MI Selector에서 k = 12, 13, 15로 했을 때 가장 좋았음. 일부 noise 를 일으키는 data를 제외해서 일반화시키기 좋음.

#### 보충해야 할 점
- 일단 데이터 셋의 Column 의 특성마다 각 Scaling 을 해줘야 했음. [참고자료](https://data-newbie.tistory.com/506)
- Column 마다 exception 을 처리할 범위가 달랐기 때문임.
- 이 또한 또 다른 smote_variants 를 처리해야 했음.
- Tabular 데이터 셋에서 적용하기 좋은 알고리즘이 XGBoost, LightGBM, CatBoost이라는 팀원의 말이 있었는데, [Gradient Boosted Tree Based Algorithm](https://medium.com/riskified-technology/xgboost-lightgbm-or-catboost-which-boosting-algorithm-should-i-use-e7fda7bb36bc) 이걸 이용을 안해봤음.
- nsml로 진행해서 ML Algorithm의 parameter를 다양하게 적용하기에 어려움이 있었음. (시간제한 및 container 활용에 따른 제약)
- 다음에는 EDA가 가능한 대회에 참여해보고 싶음. Correlation 에 따라서 다양하게 data 적용이 가능할 듯.
- Pipeline 을 좀 더 효율적으로 짜보자!!! - 귀찮고 시간없어서 안했었는데, 해보면 좋을 듯


# NSML
- NSML은 대회 진행용으로 이용할 수도 있고, 가상 환경으로서 이용할 수도 있다.
- 모델의 checkpoint 를 Save, Load를 쉽게 할 수 있다.
- Docker Container 를 올려놨다가 훈련이 끝나면 바로 docker exit. 

## NSML 이용방법
자세한 내용은 [여기](https://n-clair.github.io/ai-docs/_build/html/ko_KR/index.html) 를 참고하세요.
```bash
# 명칭이 'nia_dm_prelim'인 데이터셋을 사용해 세션 실행하기
$ nsml run -d nia_dm
# 메인 파일명이 'main.py'가 아닌 경우('-e' 옵션으로 entry point 지정)
# 예: nsml run -d nia_dm -e main_lightgbm.py
$ nsml run -d nia_dm -e [파일명]

# 세션 로그 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml logs -f [세션명]

# 세션 종료 후 모델 목록 및 제출하고자 하는 모델의 checkpoint 번호 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml model ls [세션명]

# 모델 제출 전 제출 코드에 문제가 없는지 점검하기('-t' 옵션)
$ nsml submit -t [세션명] [모델_checkpoint_번호]

# 모델 제출하기
# 제출 후 리더보드에서 점수 확인 가능
$ nsml submit [세션명] [모델_checkpoint_번호]

# session 멈추기
$ nsml stop [세션명]
```
- model checkpoint 이어서 훈련가능
- 모델의 저장은 nsml.save를 통해서 저장가능 (epoch 하나마다 저장하면 memory 많이 차지함)
