# ignore warnings

import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# import relevant libraries

import math
import pandas
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random

import itertools
from statistics import mean
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing,neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from scipy import stats

from sklearn.model_selection import KFold
from matplotlib.pyplot import figure
from sklearn.metrics import confusion_matrix

from sklearn.metrics.cluster import mutual_info_score

# compute multi-class metrics

def compute_measure(class_num, predicted_label, true_label):

    cnf_matrix = confusion_matrix(true_label, predicted_label)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # F1
    F_1 = 2 * (PPV * TPR) / (PPV + TPR)
    # Overall accuracy for each class
    ACC_Class = (TP+TN)/(TP+FP+FN+TN)
    # Average accuracy
    ACC = np.sum(np.diag(cnf_matrix)) / cnf_matrix.sum()

    d_idx = np.log2(1+ACC) + np.log2(1+ (TPR+TNR)/2)

    ans=[]
    ans.append(d_idx.mean())
    ans.append(ACC)
    ans.append(TPR[0])
    ans.append(TNR[0])
    ans.append(PPV[0])
    ans.append(NPV[0])

    return ans

# load training datasets
X_train = pd.read_csv("ovarian/ovarian_train_data.csv",index_col=0)
y_train = pd.read_csv("ovarian/ovarian_train_labels.csv",index_col=0)
y_train = y_train['file_label']
X_test = pd.read_csv("ovarian/ovarian_test_data.csv",index_col=0)
y_test = pd.read_csv("ovarian/ovarian_test_labels.csv",index_col=0)
y_test = y_test['file_label']

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
label_num = len(np.unique(y_train))

y_train.value_counts()

y_test.value_counts()

# normalization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# name list of classifiers
names = [
         "rbf-SVM",
         "RandomForest",
         "ExtraTreesClassifier",
         "NaiveBayes",
         "MLP"]


classifiers = [
        SVC(kernel='rbf'),
        RandomForestClassifier(n_estimators=500, max_depth=20),
        ExtraTreesClassifier(n_estimators=500, max_depth=20),
        GaussianNB(),
        MLPClassifier(alpha=0.0001, hidden_layer_sizes = (100, 50, 25),  max_iter=10000)]

label_names = ['class 0', 'class 1']
for name, clf in zip(names, classifiers):
    ans_ori = []
    print(name)
    clf.fit(X_train_scaled,y_train)
    test_predict_ori = clf.predict(X_test_scaled)

    print(classification_report(y_test, test_predict_ori, target_names=label_names))

