import pandas as pd
import os
import cv2
import numpy as np
pathtoframes="frames/"
directories=os.listdir(pathtoframes)
num_des=60
features=np.empty((1,num_des*64))
featurelabel=np.empty((1,1))
def extract_features(image_loc,label):
    image=cv2.imread(image_loc, cv2.IMREAD_GRAYSCALE)
    image=cv2.resize(image, (150, 150))
    surf = cv2.xfeatures2d.SURF_create()
    keypoints,descriptors = surf.detectAndCompute(image, None)
    #print(len(keypoints),descriptors.shape)
    if  len(descriptors)>=num_des:
        global features
        features=np.append(features,np.asarray(descriptors[:num_des]).reshape(1,num_des*64),axis=0)
        global featurelabel
        featurelabel=np.append(featurelabel,label)
        featurelabel.astype(int)
      #  if(l==0):
       #     print(features,featurelabel)  
     #   print(featurelabel)     
    else:
        pass
       # print("lesaa then 100 features",image_loc)
label=-1
l=0
for directory in directories:
    label+=1
    files=os.listdir(pathtoframes + directory)
    print(directory,files[0])
    l=0
    for file in files:
        extract_features(pathtoframes + directory+"/"+file,label)
        l+=1
    print(l)
features=np.delete(features,0,0)
featurelabel=np.delete(featurelabel,0,0)
features=pd.DataFrame(features)
print(features)
features.to_csv("features/surf_features_"+str(num_des)+".csv",header=False, index=False)

featurelabel=pd.DataFrame(featurelabel)

featurelabel.to_csv("features/surf_featurelabel_"+str(num_des)+".csv",header=False, index=False)