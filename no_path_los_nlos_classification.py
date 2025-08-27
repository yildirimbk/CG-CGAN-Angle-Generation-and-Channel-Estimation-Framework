### Note: print_score function is taken from:https://github.com/fares-ds/Data-Science-Projects-From-Kaggle/blob/7d71328a135cb580e2823353398579e9e025918e/predicting-heart-disease-using-machine-learning.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
import numpy as np
import os
import random
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from scipy.io import loadmat, savemat
import pickle
import joblib
import sys


#                                           ### DATA PREPARATION ###
training_data1_path = '/main/training_dataset_4_path_80percent_k1.mat' #change path with the current folder path that you are saving dataset
with h5py.File(training_data1_path, 'r') as file:
    training_data1 = file['training_data']['training_data'][:]
    train_data1 = training_data1.T
    print("Training data loaded successfully:", train_data1.shape)
val_data1_path = '/main/validation_dataset_4_path_10percent_k1.mat' #change path with the current folder path that you are saving dataset
with h5py.File(val_data1_path, 'r') as file:
    val_data1 = file['val_data']['val_data'][:]
    val_data1 = val_data1.T
    print("Validation data loaded successfully:", val_data1.shape)
test_data1_path = '/main/test_dataset_4_path_10percent_k1.mat' #change path with the current folder path that you are saving dataset
with h5py.File(test_data1_path, 'r') as file:
    test_data1 = file['test_data']['test_data'][:]
    test_data1 = test_data1.T
    print("Test data loaded successfully:", test_data1.shape)

def Custom_Dataset(dataset_type):
 
    los_status = dataset_type[:,5].astype(np.float32) #Los_status
    num_paths = dataset_type[:,6].astype(np.float32) #num_paths
    labels = dataset_type[:,0:4].astype(np.float32) #with user and BS locations
    los_status = los_status.reshape(-1,1)
    num_paths = num_paths.reshape(-1,1)
    print('los_status size:', los_status.shape)
    print('num_paths size:', num_paths.shape)
    print('Labels size:', labels.shape)

    return labels, los_status, num_paths

train_labels_o, train_los_status_o, train_num_paths_o = Custom_Dataset(train_data1)
val_labels_o, val_los_status_o, val_num_paths_o = Custom_Dataset(val_data1)
test_labels_o, test_los_status_o, test_num_paths_o = Custom_Dataset(test_data1)

s_sc_label = StandardScaler()

s_sc_label.fit(train_labels_o)
#save fitted scalers
joblib.dump(s_sc_label, "label_no_path_los_nlos_std_scl_80tr10test.pkl")
#sys.exit()
train_labels_o, val_labels_o, test_labels_o = s_sc_label.transform(train_labels_o), s_sc_label.transform(val_labels_o), s_sc_label.transform(test_labels_o)


train_dataset = np.concatenate((train_labels_o,train_los_status_o,train_num_paths_o), axis=1)
val_dataset = np.concatenate((val_labels_o,val_los_status_o,val_num_paths_o), axis=1)
test_dataset = np.concatenate((test_labels_o,test_los_status_o,test_num_paths_o), axis=1)


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# KNN Classification

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=1) #best 1( 100 | 99.8148 ) or 3 ( 99.985557 | 99.8118 ) 5 (99.940594 | 99.7826 )
knn_clf.fit(train_dataset[:,:4], train_dataset[:,4])

train_knn_features = train_dataset[:,:4].astype(np.float32)
with open("train_knn_features.npy", "wb") as f:
    np.save(f, train_knn_features)

train_knn_labels= train_dataset[:,4].astype(np.float32)
with open("train_knn_labels_los_status.npy", "wb") as f:
    np.save(f, train_knn_labels)
sys.exit()
#save fitted model
with open('knn_no_path_los_nlos_classifier_80tr10test.pkl', 'wb') as knnmodel:
    pickle.dump(knn_clf, knnmodel)


class_report = classification_report(test_dataset[:,4], knn_clf.predict(test_dataset[:,:4]), target_names=[f"Class {i}" for i in range(2)])
print(class_report)

print_score(knn_clf, train_dataset[:,:4], train_dataset[:,4], test_dataset[:,:4], test_dataset[:,4], train=True)
print_score(knn_clf, train_dataset[:,:4], train_dataset[:,4], test_dataset[:,:4], test_dataset[:,4], train=False)

# print_score(knn_clf, train_dataset[:,:5], train_dataset[:,5], val_dataset[:,:5], val_dataset[:,5], train=True)
# print_score(knn_clf, train_dataset[:,:5], train_dataset[:,5], val_dataset[:,:5], val_dataset[:,5], train=False)


test_score = accuracy_score(test_dataset[:,4], knn_clf.predict(test_dataset[:,:4])) * 100
# val_score = accuracy_score(val_dataset[:,5], knn_clf.predict(val_dataset[:,:5])) * 100
train_score = accuracy_score(train_dataset[:,4], knn_clf.predict(train_dataset[:,:4])) * 100

results_df_2 = pd.DataFrame(data=[["K-nearest neighbors", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
# results_df = pd.concat([results_df,results_df_2], ignore_index=True)
cf_matrix = confusion_matrix(test_dataset[:,4], knn_clf.predict(test_dataset[:,:4]),labels=knn_clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                              display_labels=knn_clf.classes_)

disp.plot()
plt.show()


print(results_df_2)


# #HYPERPARAMETER TUNING FOR KNN CAN BE ADDED.
# from sklearn.neighbors import KNeighborsClassifier
# train_score = []
# test_score = []
# neighbors = range(1, 10) #30

# for k in neighbors:
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(train_dataset[:,:4], train_dataset[:,4])
#     train_score.append(accuracy_score(train_dataset[:,4], model.predict(train_dataset[:,:4])))
#     test_score.append(accuracy_score(test_dataset[:,4], model.predict(test_dataset[:,:4])))

# plt.figure(figsize=(10, 7))

# plt.plot(neighbors, train_score, label="Train score")
# plt.xticks(np.arange(1, 21, 1))
# plt.xlabel("Number of neighbors")
# plt.ylabel("Model score")
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 7))
# plt.plot(neighbors, test_score, label="Test score")
# plt.xticks(np.arange(1, 21, 1))
# plt.xlabel("Number of neighbors")
# plt.ylabel("Model score")
# plt.legend()    
# plt.show()

# print(f"Maximum KNN score on the training data: {max(train_score)*100:.2f}%")
# print(f"Maximum KNN score on the test data: {max(test_score)*100:.2f}%")



