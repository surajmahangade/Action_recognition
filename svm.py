import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, GridSearchCV
#import tensorflow as tf


num_des=50
features=pd.read_csv("features/surf_features_"+str(num_des)+".csv",header=None)
featurelabel=pd.read_csv("features/surf_featurelabel_"+str(num_des)+".csv",header=None)
print(features.columns)

x_vals=features.to_numpy()
y_vals=featurelabel.to_numpy()

#config = tf.compat.v1.ConfigProto() 
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
y_vals_train=y_vals_train.flatten()
y_vals_test=y_vals_test.flatten()
svc_params = {'kernel': ['rbf'], 'gamma': [0.0001,0.001, 0.01, 0.1, 1], 'C': [0.001, 0.01, 0.1, 1, 10,100,1000]}
clf = GridSearchCV(estimator=SVC(), param_grid=svc_params, cv=20, n_jobs=-1, 
                   scoring='accuracy', verbose=10)

clf.fit(x_vals_train, y_vals_train)
y_pred = clf.predict(x_vals_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_vals_test,y_pred))
print(classification_report(y_vals_test,y_pred))
import joblib  
# Save the trained model as a pickle string. 
joblib.dump(clf, 'svm_1.pkl')