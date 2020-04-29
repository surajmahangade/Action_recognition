import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle
from sklearn.preprocessing import StandardScaler


num_des=100
features=pd.read_csv("features/surf_features_"+str(num_des)+".csv",header=None)
featurelabel=pd.read_csv("features/surf_featurelabel_"+str(num_des)+".csv",header=None)
print(features.columns)

features=features.to_numpy()
featurelabel=featurelabel.to_numpy()

total_clusters=2
from sklearn.metrics.cluster import adjusted_rand_score

import scipy
def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    for i in range(n_clusters):
        idx = labels == i
        new_label=scipy.stats.mode(real_labels[idx])[0][0]  # Choose the most common label among data points in the cluster
        permutation.append(new_label)
    return permutation


#kmeans
max_score=0
for i in range(500):
    kmeans = MiniBatchKMeans(n_clusters = total_clusters)
    features = StandardScaler().fit_transform(features)
    kmeans.fit(features)
    kmeans_pred=kmeans.predict(features)
#k-means performance:
    kmeans_pred=kmeans_pred.flatten()
    featurelabel=featurelabel.flatten()
    permutation=find_permutation(total_clusters,featurelabel,kmeans.labels_)
    new_labels = [ permutation[label] for label in kmeans.labels_]
    from sklearn.metrics import accuracy_score
    #print(accuracy_score(featurelabel, new_labels))
    #adjusted_rand_score(featurelabel, new_labels)
    if accuracy_score(featurelabel, new_labels)>max_score:
        #print(permutation)
        kmeans_saved=kmeans
        max_score=accuracy_score(featurelabel, new_labels)
        pickle.dump(kmeans_saved, open("save.pkl", "wb"))
        print("max------------------=",max_score)
    #print("kmeans =", adjusted_rand_score(featurelabel, kmeans_pred))

    
print("max------------------=",max_score)
kmeans_pred=kmeans_saved.predict(features)
temp=[]
for pred in [kmeans_pred]:
    for j in range(len(np.unique(featurelabel))):
        var=featurelabel==j
        for i in range(len(var)):
            if var[i]:
                temp.append(pred[i])
        my_dict = {i:temp.count(i) for i in temp}
        print(j)
        print(my_dict)
pickle.dump(kmeans_saved, open("save.pkl", "wb"))
#print(retrieve_info(kmeans.labels_,featurelabel))
