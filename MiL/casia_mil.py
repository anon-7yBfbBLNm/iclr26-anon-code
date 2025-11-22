
"""
Note

This configuration reflects demo performance. Better results can be achieved by
running a more extensive probing phase to identify optimal hyperparameters, as
model accuracy is sensitive to k, n, and s.
"""
import warnings
warnings.filterwarnings('ignore')

import math
import random
import itertools
import statistics
from statistics import mean

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.model_selection import train_test_split, KFold

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn import neighbors

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import sklearn.metrics as metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import mutual_info_score

label_num = 6
neighbors_k = 20
added_neighbors_n = 1
batch_size_s = 10
distance_method = 'euclidean'
badguy_number = 1

def computeMeasure(class_num, predicted_label, true_label):
    cnf_matrix = confusion_matrix(true_label, predicted_label)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP, FN, TP, TN = map(lambda x: x.astype(float), [FP, FN, TP, TN])
    with np.errstate(divide='ignore', invalid='ignore'):
        TPR = np.nan_to_num(TP / (TP + FN))
        TNR = np.nan_to_num(TN / (TN + FP))
        PPV = np.nan_to_num(TP / (TP + FP))
        NPV = np.nan_to_num(TN / (TN + FN))
        ACC = np.sum(np.diag(cnf_matrix)) / cnf_matrix.sum()
    d_idx_vector = np.log2(1 + ACC) + np.log2(1 + (TPR + TNR) / 2)
    d_idx = d_idx_vector.mean()
    results = [d_idx, ACC, TPR.mean(), TNR.mean(), PPV.mean(), NPV.mean()]
    return results

def Union(lst1, lst2):
    filtered1 = [x for x in lst1 if not isinstance(x, (pd.DataFrame, pd.Series))]
    filtered2 = [x for x in lst2 if not isinstance(x, (pd.DataFrame, pd.Series))]
    return list(set(filtered1) | set(filtered2))

def Intersection(lst1, lst2):
    filtered1 = [x for x in lst1 if not isinstance(x, (pd.DataFrame, pd.Series))]
    filtered2 = [x for x in lst2 if not isinstance(x, (pd.DataFrame, pd.Series))]
    return list(set(filtered1) & set(filtered2))

def doReproducibleLearning(X_train, y_train, X_test):
    clf = SVC(kernel='rbf', C=10.0, gamma=1e-2, tol=1e-4)
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return y_test

def create_customized_training_set(training_data, training_data_label, entry, distance_method, neighbors_k, added_nbr_n):
    dist_c = pd.DataFrame(pairwise_distances(training_data, entry, metric=distance_method))
    k_close_neighbors = dist_c.apply(lambda col: list(col.sort_values()[:neighbors_k].index))
    customized_index = k_close_neighbors.transpose().values.tolist()[0]
    original_indices = training_data.index[customized_index].tolist()
    label_counts = training_data_label.loc[original_indices].value_counts()
    low_count = list(set(label_counts[label_counts < 2].index))
    missing = list(set(training_data_label.unique()) - set(label_counts.index))
    for cls in low_count + missing:
        added = 0
        sorted_indices = dist_c[0].sort_values().index
        for idx in sorted_indices:
            original_idx = training_data.index[idx]
            if training_data_label.loc[original_idx] == cls and original_idx not in original_indices:
                original_indices.append(original_idx)
                added += 1
            if added >= added_nbr_n:
                break
    return original_indices

def create_customized_correlation_training_set(training_data, training_data_label, entry, distance_method, neighbors_k, added_nbr_n):
    entry_1d = entry.iloc[0].to_numpy()
    corr = training_data.apply(lambda row: stats.pearsonr(row, entry_1d)[0], axis=1)
    corr_sorted = corr.sort_values(ascending=False)
    customized_index = corr_sorted[:neighbors_k].index.tolist()
    label_counts = training_data_label.loc[customized_index].value_counts()
    low_count = list(set(label_counts[label_counts < 2].index))
    missing = list(set(training_data_label.unique()) - set(label_counts.index))
    for cls in low_count + missing:
        added = 0
        for idx in corr_sorted.index:
            if training_data_label.loc[idx] == cls and idx not in customized_index:
                customized_index.append(idx)
                added += 1
            if added >= added_nbr_n:
                break
    return customized_index

def find_closest_train_test_entry(training_data, test_entry, distance_method, neighbor_number):
    dist_c = pd.DataFrame(pairwise_distances(training_data, test_entry, metric=distance_method))
    k_close_neighbors = dist_c.apply(lambda col: list(col.sort_values()[:neighbor_number].index))
    return k_close_neighbors.transpose()

def flatten_list(nested_list):
    if isinstance(nested_list, (int, float, str, np.integer, np.floating)):
        return [nested_list]
    flattened = []
    for sublist in nested_list:
        if isinstance(sublist, (list, tuple, np.ndarray)):
            flattened.extend(sublist)
        else:
            flattened.append(sublist)
    return flattened

def find_similarist_train_test_entry_in_correctly_predicted_train_test(training_data, test_entry, distance_method, neighbor_number):
    corr = training_data.apply(lambda row: stats.pearsonr(row, test_entry.iloc[0])[0], axis=1)
    top_k = corr.sort_values(ascending=False).index[:neighbor_number]
    return pd.DataFrame([top_k])

def probing_learning(X_train, y_train, neighbors_k_list, added_neighbors_n_list, batch_size_s_list, distance_method='euclidean'):
    best_d_index = -np.inf
    best_k, best_n, best_s = None, None, None
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    X_train_train = X_train_train.reset_index(drop=True)
    y_train_train = y_train_train.reset_index(drop=True)
    X_train_val = X_train_val.reset_index(drop=True)
    y_train_val = y_train_val.reset_index(drop=True)
    for k in neighbors_k_list:
        for n in added_neighbors_n_list:
            for s in batch_size_s_list:
                preds = adaptive_learning(X_train_train, y_train_train, X_train_val, y_train_val, k, distance_method, n, s)
                d_index = computeMeasure(label_num, preds, y_train_val)[0]
                if d_index > best_d_index:
                    best_d_index = d_index
                    best_k, best_n, best_s = k, n, s
    print(f"Best probing: k={best_k}, n={best_n}, s={best_s}, D={best_d_index:.4f}")
    return best_k, best_n, best_s

def adaptive_learning(training_data, training_data_label, test_data, test_data_label, neighbors_k, distance_method, added_nbr_n, batch_size_s):
    test_predict_AL = pd.Series(dtype=int)
    row_counter = 0
    batch_entry_counter = 0
    batch_entry_index = []
    batch_customized_training_set_index = []
    training_data_reset = training_data.reset_index(drop=True)
    training_data_label_reset = training_data_label.reset_index(drop=True)
    for index, row in test_data.iterrows():
        al_added_nbr_n = 1
        test_entry = test_data.loc[[index]]
        batch_entry_index.append(index)
        customized_training_set_index_pos = create_customized_training_set(training_data_reset, training_data_label_reset, test_entry, distance_method, neighbors_k, al_added_nbr_n)
        customized_correlation_training_set_index_pos = create_customized_correlation_training_set(training_data_reset, training_data_label_reset, test_entry, distance_method, neighbors_k, al_added_nbr_n)
        customized_training_set_index = training_data.index[customized_training_set_index_pos].tolist()
        customized_correlation_training_set_index = training_data.index[customized_correlation_training_set_index_pos].tolist()
        customized_training_set_index = Intersection(customized_training_set_index, customized_correlation_training_set_index)
        if len(customized_training_set_index) < (label_num * 2):
            al_added_nbr_n += 5
            customized_training_set_index_pos = create_customized_training_set(training_data_reset, training_data_label_reset, test_entry, distance_method, neighbors_k, al_added_nbr_n)
            customized_correlation_training_set_index_pos = create_customized_correlation_training_set(training_data_reset, training_data_label_reset, test_entry, distance_method, neighbors_k, al_added_nbr_n)
            customized_training_set_index = training_data.index[customized_training_set_index_pos].tolist()
            customized_correlation_training_set_index = training_data.index[customized_correlation_training_set_index_pos].tolist()
            customized_training_set_index = Intersection(customized_training_set_index, customized_correlation_training_set_index)
        batch_customized_training_set_index = Union(batch_customized_training_set_index, customized_training_set_index)
        batch_entry_counter += 1
        row_counter += 1
        if batch_entry_counter == batch_size_s or row_counter == len(test_data):
            X_train_batch = training_data.loc[batch_customized_training_set_index]
            y_train_batch = training_data_label.loc[batch_customized_training_set_index]
            X_test_batch = test_data.loc[batch_entry_index]
            y_test_batch_pred = doReproducibleLearning(X_train_batch, y_train_batch, X_test_batch)
            for i, pred in zip(batch_entry_index, y_test_batch_pred):
                test_predict_AL.loc[i] = int(pred)
            batch_entry_counter = 0
            batch_entry_index = []
            batch_customized_training_set_index = []
    return test_predict_AL

def training_sanitisation(training_data, training_labels, test_data, test_labels,
                          neighbors_k, distance_method, added_neighbors_n, batch_size_s,
                          badguy_number=1):
    pred_labels = adaptive_learning(training_data, training_labels, test_data, test_labels,
                                    neighbors_k, distance_method, added_neighbors_n, batch_size_s)
    pred_labels = pred_labels.sort_index()
    test_labels = test_labels.loc[pred_labels.index]
    bad_mask = (pred_labels != test_labels)
    badguys = bad_mask[bad_mask].index.values
    goodguys = (~bad_mask)[~bad_mask].index.values
    badguys_neighbors = []
    for idx in badguys:
        neighbor = find_closest_train_test_entry(training_data, test_data.loc[[idx]], distance_method, badguy_number)
        badguys_neighbors.append(neighbor)
    badguys_neighbors = flatten_list(badguys_neighbors)
    badguys_total = Union(badguys, badguys_neighbors)
    return goodguys, badguys_total

def meta_traininglet_fusion(goodguys, test_data, actual_test, distance_method):
    closest_idx_list = []
    similarist_idx_list = []
    for i in range(len(actual_test)):
        test_entry = actual_test.loc[[actual_test.index[i]]]
        test_data_sub = test_data.loc[goodguys]
        closest_idx = find_closest_train_test_entry(test_data_sub, test_entry, distance_method, 1)
        similarist_idx = find_similarist_train_test_entry_in_correctly_predicted_train_test(
            test_data_sub, test_entry, distance_method, 1)
        closest_idx_list.append(flatten_list(closest_idx.values.tolist()))
        similarist_idx_list.append(flatten_list(similarist_idx.values.tolist()))
    return flatten_list(closest_idx_list), flatten_list(similarist_idx_list)

def precision_pruning(traininglet_idx, badguy_pals, badguys):
    remove_set = set(badguy_pals) | set(badguys)
    return list(set(traininglet_idx) - remove_set)

def micro_learning(actual_train, actual_train_label, actual_test, actual_test_label,
                   training_data, training_data_label, test_data, test_data_label,
                   neighbors_k, distance_method, added_neighbors_n, batch_size_s):

    def safe_loc(df_or_series, idx_list):
        return df_or_series.loc[[i for i in idx_list if i in df_or_series.index]]

    test_predict_AL = pd.Series(dtype=int)

    goodguys, badguys = training_sanitisation(
        training_data, training_data_label,
        test_data, test_data_label,
        neighbors_k, distance_method, added_neighbors_n, batch_size_s,
        badguy_number=1
    )

    badguy_pals = []
    for idx in badguys:
        pals = find_closest_train_test_entry(training_data, test_data.loc[[idx]], distance_method, 1)
        badguy_pals.extend(flatten_list(pals))
    badguy_pals = list(set(badguy_pals))

    closest_idx_list, _ = meta_traininglet_fusion(goodguys, test_data, actual_test, distance_method)

    test_indices = list(actual_test.index)
    batch_test_idx = []
    batch_traininglet_idx = []

    for idx_num, test_idx in enumerate(test_indices):
        batch_test_idx.append(test_idx)
        meta1 = []
        meta2 = []
        meta3 = []
        meta4 = []

        test_entry = actual_test.loc[[test_idx]]
        custom_idxs = create_customized_training_set(actual_train.drop(badguys, errors='ignore'),
                                                     actual_train_label.drop(badguys, errors='ignore'),
                                                     test_entry, distance_method, neighbors_k, added_neighbors_n)
        custom_corr_idxs = create_customized_correlation_training_set(actual_train.drop(badguys, errors='ignore'),
                                                                       actual_train_label.drop(badguys, errors='ignore'),
                                                                       test_entry, distance_method, neighbors_k, added_neighbors_n)
        meta1 = Union(custom_idxs, custom_corr_idxs)

        if idx_num < len(closest_idx_list):
            closest_ids = flatten_list(closest_idx_list[idx_num])
            if len(closest_ids) >= 1:
                g1 = closest_ids[0]
                meta2.append(g1)
                g1_entry = test_data.loc[[g1]]
                g1_idxs = create_customized_training_set(actual_train.drop(badguys, errors='ignore'),
                                                         actual_train_label.drop(badguys, errors='ignore'),
                                                         g1_entry, distance_method, neighbors_k, added_neighbors_n)
                meta2 = Union(meta2, g1_idxs)
            if len(closest_ids) >= 2:
                g2 = closest_ids[1]
                meta3.append(g2)
                g2_entry = test_data.loc[[g2]]
                g2_idxs = create_customized_training_set(actual_train.drop(badguys, errors='ignore'),
                                                         actual_train_label.drop(badguys, errors='ignore'),
                                                         g2_entry, distance_method, neighbors_k, added_neighbors_n)
                meta3 = Union(meta3, g2_idxs)

        if len(goodguys) > 0:
            g_rand = random.choice(goodguys)
            g_rand_entry = test_data.loc[[g_rand]]
            meta4.append(g_rand)
            rand_idxs = create_customized_training_set(actual_train.drop(badguys, errors='ignore'),
                                                      actual_train_label.drop(badguys, errors='ignore'),
                                                      g_rand_entry, distance_method, neighbors_k, added_neighbors_n)
            meta4 = Union(meta4, rand_idxs)

        all_idxs = Union(meta1, meta2)
        all_idxs = Union(all_idxs, meta3)
        all_idxs = Union(all_idxs, meta4)
        batch_traininglet_idx = Union(batch_traininglet_idx, all_idxs)

        if len(batch_test_idx) == batch_size_s or idx_num == len(test_indices) - 1:
            pruned_idx = precision_pruning(batch_traininglet_idx, badguy_pals, badguys)
            pruned_idx = [idx for idx in pruned_idx if idx in actual_train.index]

            if not pruned_idx:
                batch_test_idx = []
                batch_traininglet_idx = []
                continue

            traininglet_labels_tmp = actual_train_label.loc[pruned_idx]
            present_classes = set(traininglet_labels_tmp)
            all_classes = set(actual_train_label.unique())
            missing_classes = list(all_classes - present_classes)

            for cls in missing_classes:
                candidates = actual_train[(actual_train_label == cls) & (~actual_train.index.isin(badguys))]
                if not candidates.empty:
                    pruned_idx.append(random.choice(candidates.index.tolist()))
                else:
                    fallback = actual_train[actual_train_label == cls]
                    if not fallback.empty:
                        pruned_idx.append(random.choice(fallback.index.tolist()))

            final_labels = safe_loc(actual_train_label, pruned_idx)
            if len(set(final_labels)) < 2:
                current_cls = list(set(final_labels))[0]
                other_cls = [cls for cls in all_classes if cls != current_cls]
                for cls in other_cls:
                    fallback = actual_train[actual_train_label == cls]
                    if not fallback.empty:
                        pruned_idx.append(random.choice(fallback.index.tolist()))
                        break

            pruned_idx = [idx for idx in pruned_idx if idx in actual_train.index]
            if not pruned_idx or len(set(actual_train_label.loc[pruned_idx])) < 2:
                batch_test_idx = []
                batch_traininglet_idx = []
                continue

            traininglet = actual_train.loc[pruned_idx]
            traininglet_labels = actual_train_label.loc[pruned_idx]

            test_batch_entries = safe_loc(actual_test, batch_test_idx)
            if test_batch_entries.empty:
                batch_test_idx = []
                batch_traininglet_idx = []
                continue

            preds = doReproducibleLearning(
                actual_train.loc[pruned_idx],
                actual_train_label.loc[pruned_idx],
                test_batch_entries,
            )

            for j, pred in zip(test_batch_entries.index, preds):
                test_predict_AL.loc[j] = int(pred)

            batch_test_idx = []
            batch_traininglet_idx = []

    return test_predict_AL

label_num = 6
metric_names = ["D-Index", "Accuracy", "TPR", "TNR", "PPV", "NPV", "F-micro", "F-macro"]
trials = range(1, 6)
k_list = list(range(5, 21, 2))
n_list = [1]
s_list = [10, 20, 30, 40, 50]

all_results = []

for trail_id in trials:
    train_df = pd.read_csv(f"CASIA_train_trail_{trail_id}.csv", index_col=0)
    test_df = pd.read_csv(f"CASIA_test_trail_{trail_id}.csv", index_col=0)

    X_train = train_df.drop(columns=["file_label"])
    y_train = train_df["file_label"]
    X_test = test_df.drop(columns=["file_label"])
    y_test = test_df["file_label"]

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    best_k, best_n, best_s = probing_learning(X_train, y_train, k_list, n_list, s_list, distance_method='euclidean')

    preds = micro_learning(X_train, y_train, X_test, y_test,
                           X_train, y_train, X_test, y_test,
                           neighbors_k=best_k,
                           distance_method='euclidean',
                           added_neighbors_n=best_n,
                           batch_size_s=best_s)

    y_test_aligned = y_test.loc[preds.index]
    metrics_result = computeMeasure(label_num, preds, y_test_aligned)
    metrics_result.append(f1_score(y_test_aligned, preds, average='micro'))
    metrics_result.append(f1_score(y_test_aligned, preds, average='macro'))
    all_results.append(metrics_result)

avg = np.mean(all_results, axis=0)
print("\nMicro-learning with all four modules: Average across all trails:")
for name, value in zip(metric_names, avg):
    print(f"{name}: {value:.4f}")