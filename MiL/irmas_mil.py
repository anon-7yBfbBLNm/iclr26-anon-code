"""
Naive MiL is employed for this dataset, and probing-based feature selection
is used to determine the optimal subset of features before Naive MiL.
"""

import warnings
warnings.filterwarnings('ignore')

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
from sklearn import preprocessing, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, DistanceMetric
from scipy import stats
from matplotlib.pyplot import figure


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{:0.2f}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class_names = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

dataset_path = "IRMAS_all_features.csv"

if_skip_feature_selection = 0

label_num = 11

clf = SVC(kernel='rbf', C=10.0, gamma=0.01, tol=0.0001)

scaler = StandardScaler()

k_folds_k = 5

AL_distance_method = 'euclidean'

NNS_k_list = [10]

added_neighbors_n_list = [1]

batch_size_s_list = [20]

counter_num = 1


def computeDataEntropy3(x_mat):
    s = np.linalg.svd(x_mat, compute_uv=False)
    s = s / np.sum(s)
    mat_entropy = -(np.inner(s, np.log2(s)))
    return mat_entropy


def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def Intersection(lst1, lst2):
    final_list = list(set(lst1) & set(lst2))
    return final_list


def compute_measure(class_num, predicted_label, true_label):
    acc_list = []
    d_idx_list = []
    sen_list = []
    spc_list = []
    ppr_list = []
    npr_list = []
    for class_name in range(class_num):
        t_idx = predicted_label == true_label
        p_idx = class_name == true_label
        n_idx = np.logical_not(p_idx)

        tp = np.sum(np.logical_and(t_idx, p_idx))
        tn = np.sum(np.logical_and(t_idx, n_idx))

        fp = np.sum(n_idx) - tn
        fn = np.sum(p_idx) - tp

        tp_fp_tn_fn_list = np.array([tp, fp, tn, fn])

        tp = tp_fp_tn_fn_list[0]
        fp = tp_fp_tn_fn_list[1]
        tn = tp_fp_tn_fn_list[2]
        fn = tp_fp_tn_fn_list[3]

        with np.errstate(divide='ignore', invalid='ignore'):
            sen = (1.0 * tp) / (tp + fn)
            spc = (1.0 * tn) / (tn + fp)
            ppr = (1.0 * tp) / (tp + fp)
            npr = (1.0 * tn) / (tn + fn)

        acc = (tp + tn) * 1.0 / (tp + fp + tn + fn)
        d_idx = np.log2(1 + acc) + np.log2(1 + (sen + spc) / 2.0)
        acc_list.append(acc)
        d_idx_list.append(d_idx)
        sen_list.append(sen)
        spc_list.append(spc)
        ppr_list.append(ppr)
        npr_list.append(npr)

    ans = []
    ans.append(np.nanmean(acc_list))
    ans.append(np.nanmean(d_idx_list))
    ans.append(np.nanmean(sen_list))
    ans.append(np.nanmean(spc_list))
    ans.append(np.nanmean(ppr_list))
    ans.append(np.nanmean(npr_list))

    return ans


def new_compute_measure(class_num, predicted_label, true_label):
    cnf_matrix = confusion_matrix(true_label, predicted_label)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)

    ACC = np.sum(np.diag(cnf_matrix)) / cnf_matrix.sum()
    d_idx = np.log2(1 + ACC) + np.log2(1 + (TPR + TNR) / 2.0)

    ans = []
    ans.append(np.nanmean(d_idx))
    ans.append(ACC)
    ans.append(np.nanmean(TPR))
    ans.append(np.nanmean(TNR))
    ans.append(np.nanmean(PPV))
    ans.append(np.nanmean(NPV))

    return ans


def mutual_info_classif_custom(X, y):
    return mutual_info_classif(X, y, n_neighbors=5)


def probing_feature_selection(X_train, y_train, X_test, y_test):
    probing_feature_selection_scaler = MinMaxScaler()
    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(
        X_train, y_train, test_size=0.1
    )
    X_train_train_non_negative = probing_feature_selection_scaler.fit_transform(
        X_train_train
    )

    X_train_df = pd.DataFrame(data=X_train, columns=X_train.columns)
    X_test_df = pd.DataFrame(data=X_test, columns=X_test.columns)

    X_train_train_df = pd.DataFrame(data=X_train_train, columns=X_train.columns)
    X_train_test_df = pd.DataFrame(data=X_train_test, columns=X_train.columns)
    D_Index_List = []

    for i in range(5, 101, 5):
        selector = SelectKBest(
            mutual_info_classif_custom,
            k=math.ceil(X_train_train.shape[1] / 100 * i),
        )
        selector.fit(X_train_train_non_negative, y_train_train)
        cols = selector.get_support(indices=True)
        train_train_feature_selected = X_train_train_df.iloc[:, cols]
        train_test_feature_selected = X_train_test_df.iloc[:, cols]

        df_feature_selected_concated = pd.concat(
            [train_train_feature_selected, train_test_feature_selected]
        )
        df_feature_selected_concated_norm = scaler.fit_transform(
            df_feature_selected_concated
        )
        train_train_feature_selected_norm = df_feature_selected_concated_norm[
            0:train_train_feature_selected.shape[0]
        ]
        train_test_feature_selected_norm = df_feature_selected_concated_norm[
            train_train_feature_selected.shape[0]:(
                train_train_feature_selected.shape[0]
                + train_test_feature_selected.shape[0]
            )
        ]

        clf.fit(train_train_feature_selected_norm, y_train_train)
        predict_results = clf.predict(train_test_feature_selected_norm)
        ans = compute_measure(label_num, predict_results.transpose(), y_train_test)
        D_Index_List.append(ans[1])

        training_set_feature_selected = X_train_df.iloc[:, cols]
        test_set_feature_selected = X_test_df.iloc[:, cols]
        training_set_feature_selected['file_label'] = y_train
        training_set_feature_selected.to_csv(
            'data_feature_selected_training_set_' + str(i) + '%.csv'
        )
        test_set_feature_selected['file_label'] = y_test
        test_set_feature_selected.to_csv(
            'data_feature_selected_test_set_' + str(i) + '%.csv'
        )

    print("optimal_feature_pecentage:")
    print(D_Index_List.index(max(D_Index_List)) * 5 + 5)
    return D_Index_List.index(max(D_Index_List)) * 5 + 5


def create_customized_training_set(
    training_data,
    training_data_label,
    entry,
    distance_method,
    neighbors_k,
    added_nbr_n,
):
    dist = DistanceMetric.get_metric(distance_method)
    dist_c = pd.DataFrame(dist.pairwise(training_data, entry))
    k_close_neighbors = dist_c.apply(
        lambda col: list(col.sort_values()[:neighbors_k].index)
    )
    customized_training_set_index = k_close_neighbors.transpose().values.tolist()[0]

    present_classes = set(training_data_label.iloc[customized_training_set_index].unique())
    all_classes = set(training_data_label.unique())
    class_not_in_training_set = list(all_classes - present_classes)

    for class_num in class_not_in_training_set:
        added_neighbor_counter = 0
        for row in dist_c.apply(
            lambda col: list(col.sort_values().index)
        ).iterrows():
            if added_neighbor_counter == added_nbr_n:
                break
            if training_data_label.iloc[int(row[1])] == class_num:
                customized_training_set_index.append(int(row[1]))
                added_neighbor_counter += 1

    return training_data.iloc[customized_training_set_index].index


def create_customized_correlation_training_set(
    training_data,
    training_data_label,
    entry,
    distance_method,
    neighbors_k,
    added_nbr_n,
):
    corr = []
    for index, row in training_data.iterrows():
        corr.append(stats.pearsonr(row, entry.iloc[0])[0])
    corr_c = pd.DataFrame(corr)
    k_close_neighbors = corr_c.apply(
        lambda col: list(col.sort_values(ascending=False)[:neighbors_k].index)
    )
    customized_correlation_training_set_index = (
        k_close_neighbors.transpose().values.tolist()[0]
    )

    present_classes = set(training_data_label.iloc[customized_correlation_training_set_index].unique())
    all_classes = set(training_data_label.unique())
    class_not_in_training_set = list(all_classes - present_classes)

    for class_num in class_not_in_training_set:
        added_neighbor_counter = 0
        for row in corr_c.apply(
            lambda col: list(col.sort_values(ascending=False).index)
        ).iterrows():
            if added_neighbor_counter == added_nbr_n:
                break
            if training_data_label.iloc[int(row[1])] == class_num:
                customized_correlation_training_set_index.append(int(row[1]))
                added_neighbor_counter += 1

    return training_data.iloc[customized_correlation_training_set_index].index


def adaptive_learning(
    training_data,
    training_data_label,
    test_data,
    test_data_label,
    neighbors_k,
    distance_method,
    added_nbr_n,
    batch_size_s,
):
    test_predict_AL = pd.Series(dtype=float)
    batch_size = batch_size_s
    row_counter = 0
    batch_entry_counter = 0
    batch_enrty_index = []
    batch_customized_customized_training_set_index = []
    entropy_list = []
    for index, row in test_data.iterrows():
        test_entry = test_data.loc[[index]]
        batch_enrty_index.append(index)
        customized_training_set_index = create_customized_training_set(
            training_data,
            training_data_label,
            test_entry,
            distance_method,
            neighbors_k,
            added_nbr_n,
        )
        customized_correlation_training_set_index = (
            create_customized_correlation_training_set(
                training_data,
                training_data_label,
                test_entry,
                distance_method,
                neighbors_k,
                added_nbr_n,
            )
        )
        customized_training_set_index = Intersection(
            customized_training_set_index, customized_correlation_training_set_index
        )
        batch_customized_customized_training_set_index = Union(
            batch_customized_customized_training_set_index,
            customized_training_set_index,
        )
        batch_entry_counter += 1
        row_counter += 1
        if (batch_entry_counter == batch_size) or (
            row_counter == test_data.shape[0]
        ):
            present_classes = set(
                training_data_label.loc[
                    batch_customized_customized_training_set_index
                ].unique()
            )
            all_classes = set(training_data_label.unique())
            class_not_in_training_set = list(all_classes - present_classes)
            for class_num in class_not_in_training_set:
                i = random.randrange(len(training_data_label))
                while training_data_label.iloc[i] != class_num:
                    i = random.randrange(len(training_data_label))
                batch_customized_customized_training_set_index.append(i)

            batch_customized_customized_training_set = training_data.loc[
                batch_customized_customized_training_set_index
            ]
            batch_customized_customized_training_set_labels = training_data_label.loc[
                batch_customized_customized_training_set_index
            ]
            batch_entropy = computeDataEntropy3(
                batch_customized_customized_training_set
            )
            entropy_list.append(batch_entropy)
            clf.fit(
                batch_customized_customized_training_set,
                batch_customized_customized_training_set_labels,
            )
            batch_test_entry_predicted_label = clf.predict(
                test_data.loc[batch_enrty_index]
            )
            batch_test_entry_predicted_label_counter = 0
            for i in batch_enrty_index:
                test_predict_AL.loc[i] = int(
                    batch_test_entry_predicted_label[
                        batch_test_entry_predicted_label_counter
                    ]
                )
                batch_test_entry_predicted_label_counter += 1
            batch_entry_counter = 0
            batch_enrty_index = []
            batch_customized_customized_training_set_index = []
    return test_predict_AL, mean(entropy_list)


def compare_AL(
    clf,
    distance_method,
    training_data,
    training_data_label,
    test_data,
    test_data_label,
    neighbors_k=20,
    added_nbr_n=3,
    batch_size_s=10,
):
    test_predict_AL, avg_entropy = adaptive_learning(
        training_data,
        training_data_label,
        test_data,
        test_data_label,
        neighbors_k,
        distance_method,
        added_nbr_n,
        batch_size_s,
    )
    ans_AL = new_compute_measure(label_num, test_predict_AL, test_data_label)
    ans_AL.append(f1_score(test_data_label, test_predict_AL, average='micro'))
    ans_AL.append(f1_score(test_data_label, test_predict_AL, average='macro'))

    return ans_AL, test_predict_AL, test_data_label


df = pd.read_csv(dataset_path, index_col=0)
df = df.reset_index(drop=True)
X = df.drop(['file_label'], axis=1)
y = df['file_label']

kf = KFold(n_splits=k_folds_k, shuffle=True, random_state=2)
counter = 1
confusion_matrix_test_predict_AL = []
confusion_matrix_test_data_label = []

for train_index, test_index in kf.split(X):
    if counter == counter_num:
        print("Trail #" + str(counter) + ":")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if if_skip_feature_selection == 0:
            percentage_of_top_features = probing_feature_selection(
                X_train, y_train, X_test, y_test
            )

            df_train = pd.read_csv(
                "data_feature_selected_training_set_"
                + str(percentage_of_top_features)
                + "%.csv",
                index_col=0,
            )
            df_train.reset_index(drop=True, inplace=True)
            AL_X_train = df_train.drop(['file_label'], axis=1)
            AL_y_train = df_train['file_label']

            df_test = pd.read_csv(
                "data_feature_selected_test_set_"
                + str(percentage_of_top_features)
                + "%.csv",
                index_col=0,
            )
            df_test.reset_index(drop=True, inplace=True)
            AL_X_test = df_test.drop(['file_label'], axis=1)
            AL_y_test = df_test['file_label']
        elif if_skip_feature_selection == 1:
            AL_X_train = pd.DataFrame(X_train)
            AL_X_train.reset_index(drop=True, inplace=True)
            AL_y_train = y_train
            AL_y_train.reset_index(drop=True, inplace=True)
            AL_X_test = pd.DataFrame(data=X_test)
            AL_X_test.reset_index(drop=True, inplace=True)
            AL_y_test = y_test
            AL_y_test.reset_index(drop=True, inplace=True)

        scaler.fit(AL_X_train)
        AL_X_train_norm = scaler.transform(AL_X_train)
        AL_X_test_norm = scaler.transform(AL_X_test)

        AL_X_train = pd.DataFrame(
            data=AL_X_train_norm,
            columns=AL_X_train.columns,
        )
        AL_X_test = pd.DataFrame(
            data=AL_X_test_norm,
            columns=AL_X_test.columns,
        )

        max_accuracy = 0
        max_parameter_list = " "
        for NNS_k in NNS_k_list:
            for added_neighbors_n in added_neighbors_n_list:
                for batch_size_s in batch_size_s_list:
                    AL_ans_temp, temp_test_predict_AL, temp_test_data_label = (
                        compare_AL(
                            clf,
                            AL_distance_method,
                            AL_X_train,
                            AL_y_train,
                            AL_X_test,
                            AL_y_test,
                            NNS_k,
                            added_neighbors_n,
                            batch_size_s,
                        )
                    )
                    confusion_matrix_test_predict_AL.extend(
                        temp_test_predict_AL
                    )
                    confusion_matrix_test_data_label.extend(
                        temp_test_data_label
                    )
                    df_test_predict_AL = pd.DataFrame(temp_test_predict_AL)
                    df_test_predict_AL.to_csv(
                        'test_predict_AL_trail_' + str(counter) + '.csv'
                    )
                    df_test_data_label = pd.DataFrame(temp_test_data_label)
                    df_test_data_label.to_csv(
                        'test_data_label_trail_' + str(counter) + '.csv'
                    )

                    if AL_ans_temp[1] > max_accuracy:
                        max_accuracy = AL_ans_temp[1]
                        AL_ans = AL_ans_temp
                        max_parameter_list = (
                            "NNS_k: "
                            + str(NNS_k)
                            + "; added_neighbors_n: "
                            + str(added_neighbors_n)
                            + "; batch_size_s: "
                            + str(batch_size_s)
                        )
        print("AL Parameters: ")
        print(max_parameter_list)
        print("D-Index, Accuracy, TPR, TNR, PPV, NPV, F-micro, F-macro")
        print(AL_ans)
        print(" ")
    counter += 1

cnf_matrix = confusion_matrix(
    confusion_matrix_test_data_label, confusion_matrix_test_predict_AL
)
np.set_printoptions(precision=2)

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plot_confusion_matrix(
    cnf_matrix,
    classes=class_names,
    normalize=True,
    title='Normalized confusion matrix',
)

plt.show()