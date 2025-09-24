## **OFL for imbalanced learning-hard problems**
### **Dataset: Ovarian: high-dimensional extremely imbalanced data**
#####RNA_seq data with 20531 genes across 266 samples, where 4 samples are 'solid ovarian cancer samples'

### **Naive OFL is employed for this dataset**
##### Authors: Anonymous (currently)
##### (c) all right reserved


## ignore warnings

import warnings
warnings.filterwarnings('ignore')

## import relevant libraries

import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import DistanceMetric
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mutual_info_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import polynomial_kernel, sigmoid_kernel, laplacian_kernel
from scipy.spatial.distance import euclidean, cityblock, cosine, correlation
from scipy import stats
from statistics import mean
import random
import itertools
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

#######################################################################################
# Function to perform reproducible learning (classification) using the SVM
# for a single test entry or a small batch, where training data is the traininlet
# for each test entry or batch.


# To acheive the reproducible learning results: support vector machines should be used
# rather than the others
######################################################################################


def doReproducibleLearning(X_train, y_train, X_test):
    """
    Train an SVM classifier on the training data and predict the labels for the test data for reproducible learning.

    Parameters:
    - X_train: array-like or pd.DataFrame
        Feature matrix for the training data

    - y_train: array-like or pd.Series
        Labels for the training data

    - X_test: array-like or pd.DataFrame
        Feature matrix for the test data

    Returns:
    - y_test: array-like
        Predicted labels for the test data
    """

    # SVM parameters
    kernel = 'rbf'
    C = 10.0
    gamma = 1e-2
    tol = 1e-4

    # Get the configured SVM classifier
    clf = SVC(kernel=kernel, C=C, gamma=gamma, tol=tol)

    # Fit the classifier on the training data
    clf.fit(X_train, y_train)

    # Predict the labels for the test data
    y_test = clf.predict(X_test)

    return y_test

def doNormalization(X_train, X_test, normalization_bit):
    """
    Normalize the training and test feature sets using a specified normalization method.

    Parameters:
    - X_train: array-like or pd.DataFrame
        Feature matrix for the training data

    - X_test: array-like or pd.DataFrame
        Feature matrix for the test data

    - normalization_bit: int
        An integer identifier to specify the normalization method.
        1 for MinMaxScaler
        2 for StandardScaler
        3 for MaxAbsScaler
        4 for RobustScaler
        5 for QuantileTransformer
        6 for PowerTransformer

    Returns:
    - X_train_normalized: pd.DataFrame
        Normalized feature matrix for the training data

    - X_test_normalized: pd.DataFrame
        Normalized feature matrix for the test data

    Raises:
    - ValueError: If an invalid normalization_bit is provided.
    """

    # Initialize the scaler
    scaler = None

    if normalization_bit == 1:
        scaler = MinMaxScaler()
    elif normalization_bit == 2:
        scaler = StandardScaler()
    elif normalization_bit == 3:
        scaler = MaxAbsScaler()
    elif normalization_bit == 4:
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid normalization_bit. Please provide a value between 1 and 4.\n")

    # Fit the scaler on the training data and transform both the training and test data
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Convert NumPy arrays back to DataFrames
    X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns)
    X_test_normalized = pd.DataFrame(X_test_normalized, columns=X_test.columns)


    return X_train_normalized, X_test_normalized

"""### Coding parts"""

#################################################################
## Function to perform resampling based on an integer identifier
#################################################################

def doResampling(X_train, y_train, resampling_bit, random_state=42):
    """
    Perform resampling on the given training data and labels using a method specified by an integer identifier.

    Parameters:
    - X_train: array-like or DataFrame
        Training feature data.

    - y_train: array-like or Series
        Training labels.

    - resampling_bit: int
        An integer identifier to specify the resampling method.
        1 for RandomOverSampler
        2 for RandomUnderSampler
        3 for SMOTE
        4 for SMOTEENN
        5 for SMOTETomek

    - random_state: int, optional (default=42)
        Random state for reproducibility.

    Returns:
    - X_resampled: array-like or DataFrame
        Resampled training feature data.

    - y_resampled: array-like or Series
        Resampled training labels.

    Raises:
    - ValueError: If an invalid resampling_bit is provided.
    """

    # RandomOverSampler: Over-sample the minority class(es) by picking samples at random with replacement
    if resampling_bit == 1:
        resampler = RandomOverSampler(random_state=random_state)

    # RandomUnderSampler: Under-sample the majority class(es) by picking samples at random
    elif resampling_bit == 2:
        resampler = RandomUnderSampler(random_state=random_state)

    # SMOTE: Synthetic Minority Over-sampling Technique
    elif resampling_bit == 3:
        resampler = SMOTE(random_state=random_state)

    # SMOTEENN: A combination of SMOTE and Edited Nearest Neighbors (ENN)
    elif resampling_bit == 4:
        resampler = SMOTEENN(random_state=random_state)

    # SMOTETomek: A combination of SMOTE and Tomek links
    elif resampling_bit == 5:
        resampler = SMOTETomek(random_state=random_state)

    # Raise an exception if an invalid identifier is provided
    else:
        raise ValueError("Invalid resampling_bit. Please provide a value between 1 and 5.")

    # Perform resampling
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

    # Return the resampled feature data and labels
    return X_resampled, y_resampled

######################################################
## Function to perform t-SNE dimensionality reduction
######################################################

def doTSNE(data, normalization_bit=1, p_perplexity=30):
    """
    Apply t-SNE (t-Distributed Stochastic Neighbor Embedding) to reduce the dimensionality of the given data,
    optionally after normalizing it.

    Parameters:
    - data: array-like or DataFrame
        The input data to be transformed.

    - normalization_bit: int, optional (default=1)
        An integer identifier to specify the normalization method.
        1 for MinMaxScaler
        2 for StandardScaler
        3 for MaxAbsScaler
        4 for RobustScaler
        0 or any other value for no normalization

    - p_perplexity: int, optional (default=30)
        The perplexity parameter for t-SNE.
        Please adjust this parameter according to the number of observations in the data.

    Returns:
    - tsneNewData: array-like
        The transformed data in the new space.
    """

    # Normalize the data using the provided normalization bit
    data_normalized, _ = doNormalization(data, data, normalization_bit)  # Passing the same data twice to fit the existing function

    # Initialize t-SNE with the given perplexity parameter
    tsne = TSNE(perplexity=p_perplexity)

    # Fit and transform the normalized data
    tsneNewData = tsne.fit_transform(data_normalized)

    # Return the transformed data
    return tsneNewData

##########################################################
## Function to plot t-SNE results (t-SNE visualization)
##########################################################
def plotTSNE(data, normalization_bit,
             p_perplexity,
             label_column_name,
             label_num, label_names,
             legend_title,
             marker_size,
             marker_opacity):
    """
    Generate the t-SNE visualization of for input data.

    Parameters:
    - data: DataFrame
        The input data containing features and labels.

    - normalization_bit: int
        Identifier for the normalization method to apply before t-SNE.

    - p_perplexity: int
        The perplexity parameter for t-SNE that controls the beighborhood size

    - label_column_name: str
        The name of the column in 'data' that contains the labels.

    - label_num: int
        The number of unique labels.

    - label_names: list of str
        List of label names for legend.

    - legend_title: str
        The title of the legend.

    - marker_size: int
        The size of the markers in the scatter plot.

    - marker_opacity: float
        The opacity of the markers in the scatter plot.
    """

    # Obtain t-SNE results by dropping the label column and apply t-SNE
    tsne_results = doTSNE(data.drop(label_column_name, axis=1), normalization_bit, p_perplexity)

    # Add t-SNE results as new columns in the original data
    data['tsne-1'] = tsne_results[:, 0]
    data['tsne-2'] = tsne_results[:, 1]

    # Save t-SNE results to a CSV file
    df_tsne_results = pd.DataFrame(tsne_results, columns=['tsne-1', 'tsne-2'])
    filename='ovarian_tsne_results.csv'
    df_tsne_results.to_csv(filename)

    # Print out the confirmation message
    print(f"'{filename}' has been saved.")

    # Initialize the matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('$TSNE_{1}$')
    ax.set_ylabel('$TSNE_{2}$')


    tab10_colors = plt.cm.tab10.colors
    color_map = {0: tab10_colors[0], 1: tab10_colors[3]}


    # Generate the scatter plot using seaborn
    sns.scatterplot(
        x="tsne-1", y="tsne-2",
        hue=label_column_name,
        style=label_column_name,
        palette=color_map,
        data=data,
        legend="full",
        s=marker_size,
        alpha=marker_opacity,
        ax=ax
    )

    # Customize the legend
    ax.get_legend().set_title(legend_title)
    for t, l in zip(ax.get_legend().texts, label_names):
        t.set_text(l)

    # Add grid
    ax.grid(True)

    return fig

###############################################################################
## Calculate classification measures for multiclass classification
##
## The function calculates multiple metrics for classification including
## d-index, accuracy, sensitivity (TPR), specificity (TNR), precision (PPV), NPV,
## FDR, F-1 score, etc. It uses the confusion matrix as a base for calculations.
###############################################################################

def computeMeasure(class_num, predicted_label, true_label):
    """
    Compute multiple metrics for binary or multiclass classification.

    Parameters:
    - class_num: int
        The number of classes in the classification problem.

    - predicted_label: array-like
        Array of predicted labels.

    - true_label: array-like
        Array of true labels.

    Returns:
    - results: list
        List of computed metrics.
    """

    # Compute the confusion matrix
    cnf_matrix = confusion_matrix(true_label, predicted_label)

    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    # Convert to float for more precise calculations
    FP, FN, TP, TN = map(lambda x: x.astype(float), [FP, FN, TP, TN])

    # Initialize variables to store metrics
    # Use np.errstate to ignore divide by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = np.nan_to_num(TP / (TP + FN))

        # Specificity or true negative rate
        TNR = np.nan_to_num(TN / (TN + FP))

        # Precision or positive predictive value
        PPV = np.nan_to_num(TP / (TP + FP))

        # Negative predictive value
        NPV = np.nan_to_num(TN / (TN + FN))

        # False positive rate
        FPR = np.nan_to_num(FP / (FP + TN))

        # False negative rate
        FNR = np.nan_to_num(FN / (TP + FN))

        # False discovery rate
        FDR = np.nan_to_num(FP / (TP + FP))

        # F1 score
        F_1 = np.nan_to_num(2 * (PPV * TPR) / (PPV + TPR))

        # Per-class accuracy
        ACC_Class = np.nan_to_num((TP + TN) / (TP + FP + FN + TN))

        # Overall accuracy
        ACC = np.sum(np.diag(cnf_matrix)) / cnf_matrix.sum()

    # Compute discriminative power index for all classes
    d_idx_vector = np.log2(1 + ACC) + np.log2(1 + (TPR + TNR) / 2)
    d_idx = d_idx_vector.mean() # do average

    # Prepare and return results
    results = [
        d_idx,
        ACC,
        TPR.mean(),
        TNR.mean(),
        PPV.mean(),
        NPV.mean()
    ]

    return results

def kernel_mutual_information(X, Y, bandwidth=1.0):
    """
    Compute the Kernel Mutual Information (KMI) between two datasets X and Y.

    Parameters:
    - X, Y: 2D array-like, shape (n_samples, n_features)
        Input data.
    - bandwidth: float
        Bandwidth parameter for the kernel.

    Returns:
    - float
        The estimated mutual information between X and Y.
    """
    XY = np.hstack((X, Y))

    # Create a kernel density estimate for the joint distribution and the individual distributions.
    kde_XY = KernelDensity(bandwidth=bandwidth).fit(XY)
    kde_X = KernelDensity(bandwidth=bandwidth).fit(X)
    kde_Y = KernelDensity(bandwidth=bandwidth).fit(Y)

    log_XY = kde_XY.score_samples(XY)
    log_X = kde_X.score_samples(X)
    log_Y = kde_Y.score_samples(Y)

    # Compute the estimated mutual information.
    MI = np.mean(log_XY - log_X - log_Y)

    return MI

###############################################################################################
## calculate the data point similarity between a targeted test point with training data
## createCustomizedTraininglet() sub function
###############################################################################################

def calculateDataPointSimilarity(training_data, entry, distance_method_bit, custom_distance_function=None):
    """
    Calculate the metric (distance or similarity) between each row in the training data and the given entry.

    Parameters:
    - training_data: DataFrame or array-like
        The feature matrix for the training data.

    - entry: DataFrame or array-like
        a data point from testing data

    - distance_method_bit: int
        An integer identifier to specify the distance computation method.

    - custom_distance_function: callable, optional
        A custom function to compute distances when distance_method_bit = 7. The function should take in two
        arguments: the training data and the test entry, and return an array or Series of distances.

    Returns:
    - DataFrame
        A DataFrame containing the calculated metrics.

    - bool
        A boolean value indicating the sort order for the metrics.
    """

    # Define the mapping of distance_method_bit to actual distance methods
    distance_methods = {1: "euclidean", 2: "manhattan", 3: "cosine", 4: "minkowski", 5: None,
                        6: None, 7: None, 8: None, 9: None, 10: None, 11: None}

    if distance_method_bit == 5:  # Pearson correlation
        correlations = [stats.pearsonr(row, entry.iloc[0])[0] for _, row in training_data.iterrows()]
        return pd.DataFrame(correlations), False

    elif distance_method_bit == 6:  # Kernel Mutual Information
        kmi = kernel_mutual_information(training_data.values, entry.values)
        # Inverting the KMI to a distance measure.
        distances = 1 / (1 + kmi)
        return pd.DataFrame([distances] * len(training_data)), True

    elif distance_method_bit == 7:  # Kernel distance (RBF Kernel)
        kernel_matrix = rbf_kernel(training_data, entry)
        # Inverting values as higher kernel value means more similarity
        distances = 1 - kernel_matrix.diagonal()
        return pd.DataFrame(distances), True

    elif distance_method_bit == 8:  # Kernel distance (Polynomial Kernel)
        kernel_matrix = polynomial_kernel(training_data, entry)
        distances = 1 - kernel_matrix.diagonal()
        return pd.DataFrame(distances), True

    elif distance_method_bit == 9:  # Kernel distance (Sigmoid Kernel)
        kernel_matrix = sigmoid_kernel(training_data, entry)
        distances = 1 - kernel_matrix.diagonal()
        return pd.DataFrame(distances), True

    elif distance_method_bit == 10:  # Kernel distance (Laplacian Kernel)
        kernel_matrix = laplacian_kernel(training_data, entry)
        distances = 1 - kernel_matrix.diagonal()
        return pd.DataFrame(distances), True

    elif distance_method_bit == 11:  # Custom distance function
        if custom_distance_function is None:
            raise ValueError("Please provide a custom distance function for distance_method_bit=11.")
        distances = custom_distance_function(training_data, entry)
        return pd.DataFrame(distances), True

    else:
        dist = DistanceMetric.get_metric(distance_methods[distance_method_bit])
        distances = dist.pairwise(training_data, entry)
        return pd.DataFrame(distances), True

#########################################################################################
## retrieve indices of entries in the traininglet for a targeted point in testing data
## createCustomizedTraininglet() sub function
## change name NearestNeighbor...
#########################################################################################

def retrieveNeighborsIndices(metric_df, neighbors_k, sort_order):
    """
    Retrieve the indices of the k-most similar or closest neighbors based on the provided metric.

    Parameters:
    - metric_df: DataFrame
        A DataFrame containing the calculated metrics (either distances or correlations) between
        the training data and the given entry.

    - neighbors_k: int
        The number of closest or most similar neighbors to consider.

    - sort_order: bool
        A boolean value indicating the sort order for the metrics. If True, lower values are better (e.g., distance metrics).
        If False, higher values are better (e.g., Pearson correlation).

    Returns:
    - list
        A list of indices corresponding to the k-most similar or closest neighbors.
    """

    # Handle cases where lower metric values are better (e.g., distance metrics)
    if sort_order:
        # Retrieve and return the indices of the k-smallest metric values
        return metric_df.apply(lambda col: col.nsmallest(neighbors_k).index).transpose().iloc[0].tolist()

    # Handle cases where higher metric values are better (e.g., Pearson correlation)
    else:
        # Retrieve and return the indices of the k-largest metric values
        return metric_df.nlargest(neighbors_k, 0).index.tolist()

#####################################################################################################
## doTrainingRebalance
## Handle under-represented and missing labels in the traininglet by adding additional neighbors.
## almost must for multiclass
## createCustomizedTraininglet() sub function
#####################################################################################################

def doTrainingletRebalance(metric_df, closest_indices, training_data_label, added_nbr_n, sort_order):
    """
    Handle under-represented and missing labels in the traininglet by adding additional neighbors.

    Parameters:
    - metric_df: DataFrame
        A DataFrame containing the calculated metrics (either distances or correlations) between
        the training data and the given entry.

    - closest_indices: list
        A list of indices corresponding to the k-most similar or closest neighbors.

    - training_data_label: Series
        The label vector for the training data.

    - added_nbr_n: int
        The number of additional neighbors to add for each missing label.

    - sort_order: bool
        A boolean value indicating the sort order for the metrics. If True, lower values are better (e.g., distance metrics).
        If False, higher values are better (e.g., Pearson correlation).

    Returns:
    - list
        An updated list of indices corresponding to entries in the customized local training set,
        accounting for under-represented and missing labels.
    """

    # Count the occurrences of each label in the initial closest neighbors
    closest_labels_count = training_data_label.iloc[closest_indices].value_counts()

    # Identify under-represented labels (those appearing less than twice)
    under_represented_labels = set(closest_labels_count[closest_labels_count < 2].index)

    # Identify missing labels (those not appearing in the initial closest neighbors)
    missing_labels = set(training_data_label.unique()) - set(training_data_label.iloc[closest_indices].unique())

    # Handle under-represented labels
    for label in under_represented_labels:
        # Sort the metric DataFrame based on the sort order
        for idx in metric_df.sort_values(by=0, ascending=sort_order).index:
            if training_data_label.iloc[idx] == label and idx not in closest_indices:
                # Add the first occurrence of the under-represented label to closest_indices
                closest_indices.append(idx)
                break

    # Handle missing labels
    for label in missing_labels:
        added_count = 0  # Counter for the number of additional neighbors added for each missing label
        for idx in metric_df.sort_values(by=0, ascending=sort_order).index:
            # Stop adding if we have reached the maximum number of additional neighbors for this label
            if added_count == added_nbr_n:
                break
            # Check if the current index corresponds to the missing label
            if training_data_label.iloc[idx] == label:
                closest_indices.append(idx)  # Add the index to closest_indices
                added_count += 1  # Increment the counter

    return closest_indices  # Return the updated list of closest indices

##############################################################################
## createCustomizedTraininglet()
##############################################################################

def createCustomizedTraininglet(training_data, training_data_label, entry,
                                  distance_method_bit, neighbors_k, added_nbr_n):
    """
    Create a customized local training set, also known as traininglet, for a given entry.

    Parameters:
    - training_data: DataFrame
        The feature matrix for the training data.

    - training_data_label: Series
        The label vector for the training data.

    - entry: DataFrame or array-like
        The feature vector for the entry that needs a customized local training set.

    - distance_method_bit: int
        An integer identifier to specify the distance or similarity computation method.
        1 for Euclidean distance
        2 for Manhattan distance
        3 for Cosine distance
        4 for Minkowski distance
        5 for Pearson correlation

    - neighbors_k: int
        The number of closest or most similar neighbors to consider.

    - added_nbr_n: int
        The number of additional neighbors to add for each missing or under-represented label.

    Returns:
    - list
        A list of indices corresponding to the entries in the customized local training set.
    """

    # Step 1: Calculate the distance or similarity metrics between the entry and all training data.
    # `sort_order` is a boolean that specifies how to sort the metrics (True for lower is better, False for higher is better).
    metric_df, sort_order = calculateDataPointSimilarity(
        training_data, entry, distance_method_bit)

    # Step 2: Retrieve the indices of the k-most similar or closest neighbors based on the calculated metrics.
    closest_indices = retrieveNeighborsIndices(
        metric_df, neighbors_k, sort_order)

    # Step 3: Handle under-represented and missing labels by adding additional neighbors to `closest_indices`.
    final_indices = doTrainingletRebalance(
        metric_df, closest_indices, training_data_label, added_nbr_n, sort_order)

    # Return the indices corresponding to the final customized local training set.
    return training_data.iloc[final_indices].index

def createFinalTraininglet(training_data, training_data_label, test_entry, distance_method_bit, neighbors_k, added_nbr_n):
    """
    Create a combined training set for a single test entry by using both distance-based
    and correlation-based methods to select appropriate training samples.

    Parameters:
    - training_data: DataFrame
        The feature matrix for the training data.
    - training_data_label: Series
        The label vector for the training data.
    - test_entry: DataFrame
        A single row from the test data DataFrame.
    - distance_method_bit: str
        The distance method used to create training sets.
    - neighbors_k: int
        Number of neighbors for initial neighborhood search.
    - added_nbr_n: int
        Number of additional neighbors for handling data imbalance.

    Returns:
    - combined_set_index: list
        List of indices for the combined training set.
    """

    # Initial setting for the number of additional neighbors to add
    initial_added_nbr_n = 1
    # Correlation method identifier
    correlation_bit = 5
    # Threshold to check if the neighborhood size is adequate
    neighbors_size_threshold_index = 2
    # Increment for the number of added neighbors if the threshold isn't met
    added_nbr_n_increment = 5

    # Create a training set based on the distance method
    training_set_index = createCustomizedTraininglet(training_data, training_data_label, test_entry, distance_method_bit, neighbors_k, initial_added_nbr_n)
    # Create a training set based on the correlation method
    correlation_training_set_index = createCustomizedTraininglet(training_data, training_data_label, test_entry, correlation_bit, neighbors_k, initial_added_nbr_n)

    # Combine the indices from both methods
    combined_set_index = list(set(training_set_index) & set(correlation_training_set_index))

    # If the size of the combined training set is below the threshold, add more neighbors
    if len(combined_set_index) < (label_num * neighbors_size_threshold_index):
        added_nbr_n = initial_added_nbr_n + added_nbr_n_increment
        # Repeat the training set creation process with the new number of added neighbors
        training_set_index = createCustomizedTraininglet(training_data, training_data_label, test_entry, distance_method_bit, neighbors_k, added_nbr_n)
        correlation_training_set_index = createCustomizedTraininglet(training_data, training_data_label, test_entry, correlation_bit, neighbors_k, added_nbr_n)
        # Update the combined set
        combined_set_index = list(set(training_set_index) & set(correlation_training_set_index))

    return combined_set_index

def batchProcessing(training_data, training_data_label, test_data, batch_indices, batch_train_set_indices, label_num):
    """
    Processes a batch of test entries, identifies missing labels in the batch, and fills them.
    Then uses SVM to make predictions for this batch.

    Parameters:
    - training_data: DataFrame
        The feature matrix for the training data.
    - training_data_label: Series
        The label vector for the training data.
    - test_data: DataFrame
        The feature matrix for the test data.
    - batch_indices: list
        List of indices for the current batch of test data.
    - batch_train_set_indices: list
        List of indices for the training set of the current batch.
    - label_num: int
        Number of unique labels in the training_data_label.

    Returns:
    - pd.Series
        Predictions for the test entries in the current batch.
    """

    # Identify unique labels in the current batch
    unique_labels_in_batch = training_data_label.loc[batch_train_set_indices].unique()

    # Check if any labels are missing in the batch and add them if necessary
    if len(unique_labels_in_batch) != label_num:
        missing_labels = set(training_data_label.unique()) - set(unique_labels_in_batch)
        for label in missing_labels:
            samples_of_label = training_data[training_data_label == label].index.tolist()
            random_sample = random.choice(samples_of_label)
            batch_train_set_indices.append(random_sample)

    # Prepare the batch training set
    batch_train_data = training_data.loc[batch_train_set_indices]
    batch_train_labels = training_data_label.loc[batch_train_set_indices]

    # Perform ReproducibleLearning with prediction for the current batch
    predictions = doReproducibleLearning(batch_train_data, batch_train_labels, test_data.loc[batch_indices])

    return pd.Series(predictions, index=batch_indices)

def naiveOFL(training_data, training_data_label, test_data, test_data_label, neighbors_k, distance_method_bit, added_nbr_n, batch_size_s):
    """
    The main function for naive OFL.
    Orchestrates the process of creating specialized training sets for each test entry,
    processes these in batches, and makes predictions using SVM.

    Parameters:
    - training_data: DataFrame
        The feature matrix for the training data.
    - training_data_label: Series
        The label vector for the training data.
    - test_data: DataFrame
        The feature matrix for the test data.
    - test_data_label: Series
        The label vector for the test data.
    - neighbors_k: int
        Number of neighbors for initial neighborhood search.
    - distance_method_bit: str
        Method for distance calculation.
    - added_nbr_n: int
        Number of additional neighbors for handling data imbalance.
    - batch_size_s: int
        Size of the batch for OFL.

    Returns:
    - test_predict_AL: Series
        Predicted labels for the test data.
    """

    # Initialize variables
    label_num = len(training_data_label.unique())
    test_predict_naiveOFL = pd.Series(dtype=int)
    batch_indices = []
    batch_train_set_indices = []
    batch_counter = 0

    # Loop through each entry in the test data
    for index, _ in test_data.iterrows():
        # Fetch the current test entry
        test_entry = test_data.loc[[index]]
        batch_indices.append(index)

        # Get the customized training set for the current test entry
        current_train_set_indices = createFinalTraininglet(training_data, training_data_label, test_entry, distance_method_bit, neighbors_k, added_nbr_n)
        batch_train_set_indices.extend(current_train_set_indices)

        batch_counter += 1

        # If the batch is full or it's the last batch, process it
        if batch_counter == batch_size_s or index == test_data.index[-1]:
            predictions = batchProcessing(training_data, training_data_label, test_data, batch_indices, batch_train_set_indices, label_num)
            test_predict_naiveOFL = pd.concat([test_predict_naiveOFL, predictions])

            # Reset variables for the next batch
            batch_counter = 0
            batch_indices = []
            batch_train_set_indices = []

    return test_predict_naiveOFL

def loadDataset(folder_name, file_name):
    """
    Load training and test datasets.

    Parameters:
    - folder_name: str
        The name of the folder where the data files are stored.
    - file_name: str
        The base name of the data files (excluding '_train.csv', '_test.csv', etc.)

    Returns:
    - AL_X_train: DataFrame
        The feature matrix for the training data.
    - AL_y_train: Series
        The label vector for the training data.
    - AL_X_test: DataFrame
        The feature matrix for the test data.
    - AL_y_test: Series
        The label vector for the test data.
    """

    # Read combined training data
    train_data = pd.read_csv(f"{folder_name}/{file_name}_train.csv", index_col=0)
    AL_X_train = train_data.drop(columns='file_label')
    AL_y_train = train_data['file_label']

    # Read combined test data
    test_data = pd.read_csv(f"{folder_name}/{file_name}_test.csv", index_col=0)
    AL_X_test = test_data.drop(columns='file_label')
    AL_y_test = test_data['file_label']

    return AL_X_train, AL_y_train, AL_X_test, AL_y_test

## 'naive_OFL_results_' + str(data_name) + '.txt'

def saveNaiveOFLResults2File(parameters_str, classification_rep, compute_measures_str, computed_measures_results):
    """
    Save naive OFL results to a text file.

    Parameters:
    - parameters_str: str
        String representation of the naive OFL parameters.

    - classification_rep: str
        Classification report string.

    - compute_measures_str: str
        Computed measures description string.

    - computed_measures_results: list or str
        Computed measures results.

    Returns:
    None. Writes to a file.
    """

    # Format the list of computed measures to a string if compute_measure returns a list.
    if isinstance(computed_measures_results, list):
        computed_measures_results = ', '.join(map(str, computed_measures_results))

    # Writing the results to a text file
    with open('naive_OFL_results.txt', 'w') as f:
        f.write("Naive OFL Results\n")
        f.write("=" * 50 + '\n')
        f.write("OFL Parameters: \n")
        f.write(parameters_str + '\n')
        f.write("=" * 50 + '\n')
        f.write("Classification Report: \n")
        f.write(classification_rep + '\n')
        f.write("=" * 50 + '\n')
        f.write(compute_measures_str + '\n')
        f.write(str(computed_measures_results) + '\n')

    print("Results have been saved to 'naive_OFL_results.txt'")

def doNaiveOFLResultsRetrieval(filename='naive_OFL_results.txt'):
    """
    Retrieve the contents of the naive OFL results file and print them.

    Parameters:
    - filename: str
        Name of the file to retrieve the results from. Default is 'naive_OFL_results.txt'.

    Returns:
    None. Prints the results.
    """

    # Reading the results back from the text file to print them out
    with open(filename, 'r') as f:
        saved_results = f.read()

    print(f"Contents of '{filename}':")
    print(saved_results)

def setNaiveOFLParameters():
    """
    Define and return parameters for naive OFL.

    Returns:
    - Dictionary containing parameters for naive OFL.
    """
    params = {
        'num_nearest_neighbors': 10,
        'num_added_neighbors': 1,
        'batch_size': 1,
        'OFL_distance_method': 1
    }

    return params

## demoNaiveOFLLearning

def doNaiveOFLLearning(OFL_X_train, OFL_y_train, OFL_X_test, OFL_y_test, parameters):
    """
    Execute naive OFL with the given parameters and return results.

    Returns:
    - Dictionary containing naive OFL results, parameters string, classification report, computed measures string, and computed measures results.
    """
    # Extracting parameters from the dictionary
    num_nearest_neighbors = parameters['num_nearest_neighbors']
    num_added_neighbors = parameters['num_added_neighbors']
    batch_size = parameters['batch_size']
    OFL_distance_method = parameters['OFL_distance_method']

    # Run the naive OFL algorithm and obtain predictions
    test_predict_OFL = naiveOFL(OFL_X_train, OFL_y_train, OFL_X_test, OFL_y_test, num_nearest_neighbors, OFL_distance_method, num_added_neighbors, batch_size)

    # Save the predictions to a CSV file
    save_path = "ovarian/ovarian_test_predict_OFL.csv"
    test_predict_OFL.to_csv(save_path)

    # Check if file was saved and print a confirmation
    if os.path.exists(save_path):
        print(f"Predictions saved successfully to {save_path}")
    else:
        print(f"Error: Predictions were not saved to {save_path}")

    # Initialize a list to store various performance measures
    ans_OFL = []

    # Compute and append performance measures
    label_num = len(OFL_y_train.unique())
    ans_OFL.extend(computeMeasure(label_num, test_predict_OFL, OFL_y_test))  # Assume compute_measure is a function that computes various metrics

    # Compute and append F1-scores (micro and macro)
    ans_OFL.append(f1_score(OFL_y_test, test_predict_OFL, average='micro'))
    ans_OFL.append(f1_score(OFL_y_test, test_predict_OFL, average='macro'))

    results = {
        'parameters_str': f"NNS_k: {num_nearest_neighbors}; added_neighbors_n: {num_added_neighbors}; batch_size_s: {batch_size}",
        'classification_rep': classification_report(OFL_y_test, test_predict_OFL, target_names=label_names),
        'compute_measures_str': "D-Index, Accuracy, TPR, TNR, PPV, NPV, F-micro, F-macro",
        'computed_measures_results': ans_OFL
    }

    return results

## It assume data is in a folder called ovarian

folder_name = file_name = "ovarian"
AL_X_train, AL_y_train, AL_X_test, AL_y_test = loadDataset(folder_name, file_name)

## Visualize training datasets

normalizaion_bit = 1
label_column_name = "file_label"
label_num = len(AL_y_train.unique())
label_names = ['recurrent','solid']
legend_title = "Ovarian tumor"

AL_Xy_train = AL_X_train.copy()
AL_Xy_train['file_label'] = AL_y_train

fig = plotTSNE(AL_Xy_train, normalizaion_bit, 10, label_column_name, label_num, label_names, legend_title, 100, 0.9)
fig.show()

## Resampling: doResampling
resampling_bit = 1
AL_X_train, AL_y_train = doResampling(AL_X_train, AL_y_train, 1)

## Normalize datasets

normalizaion_bit = 2
AL_X_train, AL_X_test = doNormalization(AL_X_train, AL_X_test, 2)

#######################################################################
## Define parameters for Naive OFL
## and apply Naive OFL
######################################################################

params = setNaiveOFLParameters()
naive_OFL_results = doNaiveOFLLearning(AL_X_train, AL_y_train, AL_X_test, AL_y_test, params)

## Save Naive OFL Results to File

saveNaiveOFLResults2File(
    naive_OFL_results['parameters_str'],
    naive_OFL_results['classification_rep'],
    naive_OFL_results['compute_measures_str'],
    naive_OFL_results['computed_measures_results']
)

## Reading the results back from the text file to print them out

doNaiveOFLResultsRetrieval()