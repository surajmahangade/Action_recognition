from joblib import dump, load
import pandas as pd
import scipy
import cv2
from sklearn.preprocessing import StandardScaler
num_des=100
features=pd.read_csv("features/surf_features_"+str(num_des)+".csv",header=None)
featurelabel=pd.read_csv("features/surf_featurelabel_"+str(num_des)+".csv",header=None)
features=features.to_numpy()
#not for svm ---features = StandardScaler().fit_transform(features)
featurelabel=featurelabel.to_numpy()
featurelabel=featurelabel.flatten()
total_clusters=2
def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    for i in range(n_clusters):
        idx = labels == i
        new_label=scipy.stats.mode(real_labels[idx])[0][0]  # Choose the most common label among data points in the cluster
        permutation.append(new_label)
    return permutation

clf = load('save.pkl')
permutation=find_permutation(total_clusters,featurelabel,clf.labels_)
print(permutation)
img=cv2.imread("jogging.jpg")
surf = cv2.xfeatures2d.SURF_create()
keypoints,descriptors = surf.detectAndCompute(img, None)
descriptors = StandardScaler().fit_transform(descriptors)
pred=clf.predict(descriptors[:num_des].reshape(1,-1))
print(pred)
c=0
cap=cv2.VideoCapture(0)
surf = cv2.xfeatures2d.SURF_create(500)
while(True):
    ret,frame =cap.read()
    if ret:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame=cv2.resize(frame, (150, 150))
        keypoints,descriptors = surf.detectAndCompute(frame, None)
        descriptors = StandardScaler().fit_transform(descriptors)
        c+=1
        if c<5 or len(descriptors)<num_des:
            continue

        pred=clf.predict(descriptors[:num_des].reshape(1,-1))
        #print(pred)
        print(permutation[pred[0]])
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
# print(clf.labels_)

cap.release()
cv2.destroyAllWindows()