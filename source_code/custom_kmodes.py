import pandas as pd
import random
from collections import Counter

# initialize custom kmodes class
class k_modes:
    # method
    # initialize model
    def __init__(self, n_clusters=3, n_init=10, max_iter=300, random_state=2019):
        self.n_clusters = n_clusters
        self.n_init=n_init
        self.max_iter=max_iter
        self.random_state=random_state
        self.labels_=list()
        self.predict=list()
        self.purity=0
        self.cluster_counter=Counter()
        self.cluster_centers=list()
        self.n_iter=0
    # method
    # train the model using dataset
    def fit(self, dataset):

        len1 = dataset.shape[0]
        feature_len = dataset.shape[1]

        centroid_index = []
        centroid_index2=[]
        centroid_width=[]
        # 1. choose best initial centroid value using n_init, random_state
        for j in range(0,self.n_init):
            for i in range(0, self.n_clusters):
                if j==0 and i==0:
                    random.seed(self.random_state)
                randnum = random.randint(0, len1)
                centroid_index.append(randnum)
                pass
            centroid_index.sort()
            # 1.1 calculate the maximum index length of picked centroids sets
            centroid_width.append(max(centroid_index) - min(centroid_index))
            centroid_index2.append(centroid_index)
            centroid_index=[]
            pass
        # 1.2 choose initial centroids sets which has the maximum index length
        final_index=centroid_width.index(max(centroid_width))
        centroid_index=centroid_index2[final_index]
        centroid_len = len(centroid_index)

        print("Initialized Centroid Index: ", centroid_index)
        print("n_clusters: ", centroid_len)
        predict = []
        count = [0] * centroid_len

        print("********** iter 1 start ************")
        # calculate distance with mode vectors using hamming distance policy
        for i in range(0, len1):
            count = [0] * centroid_len
            for j in range(0, centroid_len):
                for k in range(0, feature_len):
                    # if comparing 2 values are different, distance=distance+1
                    if dataset.iloc[centroid_index[j], k] != dataset.iloc[i, k]:
                        count[j] = count[j] + 1
                        pass
                    pass
                pass
            # check which centroid has the minimum distance value
            centroid_result = count.index(min(count))
            # clustering
            predict.append(centroid_result)
            pass
        print("********** iter 1 end ************")
        # print clustering result
        print(Counter(predict))
        dataset['cluster'] = predict

        # iteration >= 2
        iter_num = 2
        while True:
            # flag for stop iteration which there's no movement of clusters change
            stop_flag = 0
            new_centroid_list = []
            new_centroid_value = []
            print("********** Computing the new center ************")
            # find the new centroid(mode vector) for cluster
            for k in range(0, centroid_len):
                df2 = dataset[dataset['cluster'] == k]
                df2_len = df2.shape[0]
                # if cluster size is 0, continue
                if df2_len == 0:
                    continue
                # new centroid is the vector which has the most frequent values in each features
                # use mode() to find the most frequent values
                new_centroid = df2.mode().loc[0]
                # add new centroid
                new_centroid_list.append(new_centroid)
                # pre-calculate a distance if a centroid & new data has same vector value
                list2 = []
                for i in range(0, feature_len):
                    num = df2.iloc[:, i].value_counts().values[0]
                    num = 1 - (num / df2_len)
                    list2.append(num)
                    pass
                new_centroid_value.append(list2)
                pass
            # print("n_clusters: ", len(new_centroid_list))
            # new iter start
            predict = []
            print("********** iter ", iter_num, " start ************")
            for i in range(0, len1):
                count = [0] * len(new_centroid_list)
                for j in range(0, len(new_centroid_list)):
                    for k in range(0, feature_len):
                        if new_centroid_list[j][k] != dataset.iloc[i, k]:
                            count[j] = count[j] + 1
                            pass
                        # if centroid's feature's value == dataset's feature's value
                        elif new_centroid_list[j][k] == dataset.iloc[i, k]:
                            # add the pre-calculated value
                            count[j] = count[j] + new_centroid_value[j][k]
                            pass
                        pass
                    pass
                centroid_result = count.index(min(count))
                # print(centroid_result, " // ", dataset.iloc[i, feature_len])
                if centroid_result != dataset.iloc[i, feature_len]:
                    # if there's no cluster movement, set stop flag
                    stop_flag = 1
                    pass
                predict.append(centroid_result)
                pass
            dataset['cluster'] = predict
            print("********** iter ", iter_num, " end ************")
            print(Counter(predict))
            # save attributes
            self.predict=predict
            self.labels_=predict
            self.cluster_counter=Counter(predict)
            iter_num = iter_num + 1
            # if stop flag is 1, or over max_iter, stop the training
            if stop_flag == 0 or iter_num+1==self.max_iter+1:
                self.n_iter=iter_num
                self.cluster_centers=new_centroid_list
                print("********* Finish **********")
                break
            pass
        pass
    # methood
    # return predict cluster list
    def predict(self):
        return self.predict
    # method
    # return purity value
    def cal_purity(self, predict, y):
        predict = pd.DataFrame(predict, columns=['cluster'])
        y = pd.DataFrame(y)
        r = pd.concat([predict, y], axis=1)

        list_key = list(Counter(predict['cluster'].values).keys())
        list_value = list(Counter(predict['cluster'].values).values())
        r = r.sort_values(['cluster'], ascending=True)
        dic = dict(zip(list_key, list_value))

        list_key2 = list(dic.keys())
        list_value2 = list(dic.values())

        poison_sum = r.shape[0]
        purity = 0

        for n in range(0, len(list_key2)):
            poison1 = r.loc[(r['cluster'] == n) & (r['class'] == 'p')].shape[0]
            poison2 = r.loc[(r['cluster'] == n) & (r['class'] == 'e')].shape[0]
            poison_check = max(poison1, poison2)
            poison_check = poison_check / poison_sum
            purity = purity + poison_check
            pass
        return purity
        pass
    pass







