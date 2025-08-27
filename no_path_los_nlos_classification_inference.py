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

val_labels_o, val_los_status_o, val_num_paths_o = Custom_Dataset(val_data1)
test_labels_o, test_los_status_o, test_num_paths_o = Custom_Dataset(test_data1)

#load fitted scalers

label_scaler=joblib.load('label_no_path_los_nlos_std_scl_80tr10test.pkl')


val_labels_o, test_labels_o = label_scaler.transform(val_labels_o), label_scaler.transform(test_labels_o)


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


# KNN Classification Inference

from sklearn.neighbors import KNeighborsClassifier
#load knn_model
with open('knn_no_path_los_nlos_classifier_80tr10test.pkl', 'rb') as knnmodel:
    knn_clf = pickle.load(knnmodel)

print("Algorithm used:", knn_clf._fit_method)
#sys.exit()

test_score = accuracy_score(test_dataset[:,4], knn_clf.predict(test_dataset[:,:4])) * 100
# val_score = accuracy_score(val_dataset[:,4], knn_clf.predict(val_dataset[:,:4])) * 100
print(test_score)
# print(val_score)


cf_matrix_test = confusion_matrix(test_dataset[:,4], knn_clf.predict(test_dataset[:,:4]),labels=knn_clf.classes_)
# cf_matrix_val = confusion_matrix(val_dataset[:,4], knn_clf.predict(val_dataset[:,:4]),labels=knn_clf.classes_)

# cf_matrix_test_percent = cf_matrix_test.astype('float') / np.sum(cf_matrix_test) * 100

# # Display normalized confusion matrix
# disp_test = ConfusionMatrixDisplay(confusion_matrix=cf_matrix_test_percent,
#                                    display_labels=knn_clf.classes_)

# fig, ax = plt.subplots()
# disp_test.plot(cmap='viridis', ax=ax, colorbar=True)

# # Rescale colorbar to 0?100%
# im = ax.images[0]
# im.set_clim(0, 100)

# # Optionally, format text labels to show % with 2 decimal
# for row in disp_test.text_:
#     for text in row:
#         value = text.get_text()
#         try:
#             val_float = float(value)
#             text.set_text(f"{val_float:.1f}%")
#         except ValueError:
#             pass
# plt.savefig("knn_confusion_matrix_percent_viridis.eps", format='eps', bbox_inches='tight')

# Row-wise normalization for color
cf_matrix_normalized = cf_matrix_test.astype('float') / cf_matrix_test.sum(axis=1, keepdims=True) * 100
cf_matrix_normalized = np.nan_to_num(cf_matrix_normalized)  # handle zero-division

disp = ConfusionMatrixDisplay(
    confusion_matrix=cf_matrix_normalized,
    display_labels=knn_clf.classes_
)

fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(cmap='viridis', ax=ax, colorbar=True)

# Set colorbar range
ax.images[0].set_clim(0, 100)

# Show raw counts as text in each cell
for (i, j), val in np.ndenumerate(cf_matrix_test):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=9)

plt.savefig("knn_confusion_matrix_row_norm_percent_viridis.eps", format='eps', bbox_inches='tight')


plt.show()



#plt.close()


#disp_test = ConfusionMatrixDisplay(confusion_matrix=cf_matrix_test,
#                              display_labels=knn_clf.classes_)
# disp_val = ConfusionMatrixDisplay(confusion_matrix=cf_matrix_val,
                            #    display_labels=knn_clf.classes_)

#disp_test.plot() #viridis default
#Save the plot as an .eps file
#plt.savefig('confusion_matrix_test_knn_cividis.eps', format='eps')
#plt.show()


#disp_test.plot()
# disp_val.plot()

#plt.show()


