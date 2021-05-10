# custom_kmodes
custom k-modes algorithm

군집화 알고리즘 중 하나인 k-modes 분류를 구현해 라이브러리로 사용할 수 있게 만든 코드입니다.<br/><br/>
자세한 내용은 링크(https://drive.google.com/file/d/1vOa3gB7Ym1rGthQDyNyixlUlHZ9g2AVQ/view?usp=sharing) 를 참조해주시기 바랍니다.

# 사용법

```python
import pandas as pd
from k_modes.custom_kmodes import k_modes

# read dataset
dataset=pd.read_csv("mushrooms.csv")
# drop target variable for clustering
X = dataset.drop(columns=["class"])
y = dataset["class"]
# initialize model
model=k_modes(n_clusters=8, n_init=3, max_iter=300, random_state=2019)
# training
model.fit(X)
# return clustering result
cluster_status=model.cluster_counter
centroid_list=model.cluster_centers
purity=model.cal_purity(model.labels_, y)
# print result
print("Result")
print("Cluster status: ",cluster_status)
print("Purity: ",purity)
```
# 결과
```python
********** iter  11  start ************
********** iter  11  end ************
Counter({0: 3875, 2: 2769, 1: 1480})
********** Computing the new center ************
********** iter  12  start ************
********** iter  12  end ************
Counter({0: 3875, 2: 2769, 1: 1480})
********* Finish **********
Result
Cluster status:  Counter({0: 3875, 2: 2769, 1: 1480})
Purity:  0.7581240768094535

Process finished with exit code 0
```

