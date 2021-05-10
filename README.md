# custom_kmodes

군집화 알고리즘 중 하나인 k-modes 분류를 구현해 라이브러리로 사용할 수 있게 만든 코드입니다.<br/><br/>
k-modes 알고리즘은 [Categorical Value](https://en.wikipedia.org/wiki/Categorical_variable, "categorical value")를 군집화할 때 사용합니다.<br/>
Python의 대표적인 머신러닝 라이브러리인 [scikit-learn](https://scikit-learn.org/stable, "scikit-learn") 스타일과 유사하게 사용할 수 있게 작성했습니다.<br/>
자세한 내용은 [링크](https://drive.google.com/file/d/1vOa3gB7Ym1rGthQDyNyixlUlHZ9g2AVQ/view?usp=sharing, "링크")를 참조해주시기 바랍니다.<br/>

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
## n_clusters=3
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

## n_clusters=4
```python
********** iter  8  start ************
********** iter  8  end ************
Counter({1: 3596, 3: 1970, 2: 1371, 0: 1187})
********** Computing the new center ************
********** iter  9  start ************
********** iter  9  end ************
Counter({1: 3596, 3: 1970, 2: 1371, 0: 1187})
********* Finish **********
Result
Cluster status:  Counter({1: 3596, 3: 1970, 2: 1371, 0: 1187})
Purity:  0.870384047267356

Process finished with exit code 0
```

## n_clusters=8
```python
********** iter  16  start ************
********** iter  16  end ************
Counter({2: 2233, 4: 1336, 3: 1205, 6: 961, 5: 897, 1: 786, 0: 514, 7: 192})
********** Computing the new center ************
********** iter  17  start ************
********** iter  17  end ************
Counter({2: 2233, 4: 1336, 3: 1205, 6: 961, 5: 897, 1: 786, 0: 514, 7: 192})
********* Finish **********
Result
Cluster status:  Counter({2: 2233, 4: 1336, 3: 1205, 6: 961, 5: 897, 1: 786, 0: 514, 7: 192})
Purity:  0.8879862136878385

Process finished with exit code 0
```

