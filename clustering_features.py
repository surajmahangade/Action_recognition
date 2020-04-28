import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
import scipy


def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = features[permutation]
    shuffled_b = featurelabel[permutation]
    return shuffled_a, shuffled_b

def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    for i in range(n_clusters):
        idx = labels == i
        new_label=scipy.stats.mode(real_labels[idx])[0][0]  # Choose the most common label among data points in the cluster
        permutation.append(new_label)
    return permutation

features=pd.read_csv("features/surf_features_60.csv",header=None)
featurelabel=pd.read_csv("features/surf_featurelabel_60.csv",header=None)
print(features.columns)

features=features.to_numpy()
featurelabel=featurelabel.to_numpy()
features,featurelabel=randomize(features,featurelabel)

total_clusters=20


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

    if adjusted_rand_score(featurelabel, new_labels)>max_score:
        print(permutation)
        kmeans_saved=kmeans
        max_score=adjusted_rand_score(featurelabel, new_labels)
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
