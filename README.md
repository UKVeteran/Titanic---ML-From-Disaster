# ⚓ TITANIC - ML From Disaster ⚓

Link To The Kaggle Dataset: <a href="https://www.kaggle.com/competitions/titanic/data"> Titanic Dataset </a>


# 1️⃣ Overview

The data has been split into two groups:

<br> 
training set (train.csv)<br> 

test set (test.csv)<br> 
<br> 
The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.


# 2️⃣ The Task

The goal is to create a predictive model that accurately determines whether a given passenger would survive the Titanic disaster based on these attributes.

# 3️⃣ Data Vizualization

![Survived](https://github.com/user-attachments/assets/2d5191b2-a198-4260-8579-b1a3258ef1a1)
![Sex](https://github.com/user-attachments/assets/194b6ea1-c113-4f72-8983-8fac905f8faf)
![Pclass](https://github.com/user-attachments/assets/4b39527e-8a98-44f8-85c0-2f84a3a07471)
![Embarked](https://github.com/user-attachments/assets/81726e4e-b244-4485-9548-d4e9d64f0175)
![SibSp](https://github.com/user-attachments/assets/0af088f7-b83d-44c8-8a21-7cf264dce634)
![Parch](https://github.com/user-attachments/assets/f196ddc9-f46d-4b03-8270-8d2341c35b02)

## The Heatmap
![Heatmap](https://github.com/user-attachments/assets/a0d2efed-1768-42d5-9ce6-ff68dd6ccd3d)


# 4️⃣ ML Models

```python
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

```









# 5️⃣ Pipelining

## Setting The Stage

### Numerical Pipe
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
numerical_pipe = Pipeline([
 ('imputer',SimpleImputer(strategy='mean')),
 ('scaler', RobustScaler())
])
numerical_pipe
```

### Categorical Pipe
```python
from sklearn.preprocessing import OneHotEncoder
cat_pipe = Pipeline([
 ('imputer',SimpleImputer(strategy='most_frequent')),
 ('encoder', OneHotEncoder(sparse=False,
 drop='if_binary',
 handle_unknown = 'ignore'))
])
cat_pipe
```

### Transformers

```python
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
num_selector = make_column_selector(dtype_include=['float','int'])
cat_selector = make_column_selector(dtype_include=['object'])
preprocessing_pipe = ColumnTransformer([
 ('numerical', numerical_pipe, num_selector),
 ('categorical', cat_pipe, cat_selector)
])
preprocessing_pipe
```


## SVC
```python

model_pipe = Pipeline([
 ('preprocessing', preprocessing_pipe),
 ('model', SVC())
])
model_pipe

```
![SVC_pipeline](https://github.com/user-attachments/assets/c1c4062b-7765-434f-86c2-78ef114faa5e)

## LR

```python

model_pipe1 = Pipeline([
 ('preprocessing', preprocessing_pipe),
 ('model', LogisticRegression())
])
model_pipe1

```

![LR_pipeline](https://github.com/user-attachments/assets/7979f5c9-e086-4d7e-bd49-adc603beb682)

