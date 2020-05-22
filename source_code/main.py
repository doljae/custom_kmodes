import pandas as pd
# import custom kmodes library
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
