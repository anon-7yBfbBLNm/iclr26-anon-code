import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

def doPCA(data, normalizaion_bit=1):
  if normalizaion_bit==1:
    normalized_data=MinMaxScaler().fit_transform(data)
  elif normalizaion_bit==2:
    normalized_data=StandardScaler().fit_transform(data)
  else:
    normalized_data=data
  pca = PCA(n_components=2)
  pcaNewData = pca.fit_transform(normalized_data)
  return pcaNewData

def plotPCA(data, normalizaion_bit, label_column_name, label_num, label_names, legend_title, marker_size, marker_opacity):
    pca_results = doPCA(data.drop(label_column_name,1), normalizaion_bit)
    data['pca-1'] = pca_results[:,0]
    data['pca-2'] = pca_results[:,1]
    fig = plt.figure(figsize=(8,8))
    fig = sns.set_theme()
    fig = plt.xlabel('$PCA_{1}$')
    fig = plt.ylabel('$PCA_{2}$')
    fig = sns.scatterplot(
        x="pca-1", y="pca-2",
        hue=label_column_name,
        style=label_column_name,
        palette=sns.color_palette("tab10",n_colors=label_num),
        data=data,
        legend="full",
        s = marker_size,
        alpha= marker_opacity
    )
    fig.get_legend().set_title(legend_title)
    for t, l in zip(fig.get_legend().texts, label_names): t.set_text(l)

def doTSNE(data, normalizaion_bit=1, p_perplexity = 100):
  if normalizaion_bit==1:
    normalized_data=MinMaxScaler().fit_transform(data)
  elif normalizaion_bit==2:
    normalized_data=StandardScaler().fit_transform(data)
  else:
    normalized_data=data
  tsne = TSNE(perplexity = p_perplexity)
  tsneNewData = tsne.fit_transform(normalized_data)
  return tsneNewData

def plotTSNE(data, normalizaion_bit, p_perplexity, label_column_name, label_num, label_names,
             legend_title, marker_size, marker_opacity):
    tsne_results = doTSNE(data.drop(label_column_name,1), normalizaion_bit, p_perplexity)
    data['tsne-1'] = tsne_results[:,0]
    data['tsne-2'] = tsne_results[:,1]
    fig = plt.figure(figsize=(8,8))
    fig = sns.set_theme()
    fig = plt.xlabel('$TSNE_{1}$')
    fig = plt.ylabel('$TSNE_{2}$')
    fig = sns.scatterplot(
        x="tsne-1", y="tsne-2",
        hue=label_column_name,
        style=label_column_name,
        palette=sns.color_palette("tab10",n_colors=label_num),
        data=data,
        legend="full",
        s = marker_size,
        alpha= marker_opacity
    )
    fig.get_legend().set_title(legend_title)
    for t, l in zip(fig.get_legend().texts, label_names): t.set_text(l)

def doUMAP(data, normalizaion_bit=1, n_neighbors=30, min_dist=0.05,
            metric='euclidean', init='random'):
  if normalizaion_bit==1:
    normalized_data=MinMaxScaler().fit_transform(data)
  elif normalizaion_bit==2:
    normalized_data=StandardScaler().fit_transform(data)
  else:
    normalized_data=data

  umap_ = umap.UMAP(n_neighbors = n_neighbors,
                      min_dist = min_dist,
                      metric = metric,
                      init = init)

  umapNewData = umap_.fit_transform(normalized_data)
  return umapNewData

def plotUMAP(data, normalizaion_bit, n_neighbors, min_dist, metric, init, label_column_name, label_num, label_names,
             legend_title, marker_size, marker_opacity):
    umap_results = doUMAP(data.drop(label_column_name,1), normalizaion_bit, n_neighbors, min_dist, metric, init)
    data['umap-1'] = umap_results[:,0]
    data['umap-2'] = umap_results[:,1]
    fig = plt.figure(figsize=(8,8))
    fig = sns.set_theme()
    fig = plt.xlabel('$UMAP_{1}$')
    fig = plt.ylabel('$UMAP_{2}$')
    fig = sns.scatterplot(
        x="umap-1", y="umap-2",
        hue=label_column_name,
        style=label_column_name,
        palette=sns.color_palette("tab10",n_colors=label_num),
        data=data,
        legend="full",
        s = marker_size,
        alpha= marker_opacity
    )
    fig.get_legend().set_title(legend_title)
    for t, l in zip(fig.get_legend().texts, label_names): t.set_text(l)

def evaluateClustering(labels_true, labels, test_data):

  homogeneity_score= metrics.homogeneity_score(labels_true, labels)
  print("Homogeneity: %0.3f" %homogeneity_score)

  completeness_score= metrics.completeness_score(labels_true, labels)
  print("Completeness: %0.3f" %completeness_score)

  v_measure_score= metrics.v_measure_score(labels_true, labels)
  print("V-measure: %0.3f" %v_measure_score)

  rand_score = metrics.adjusted_rand_score(labels_true, labels)
  print("Adjusted Rand Index: %0.3f"% rand_score)

  mutual_info_score=metrics.adjusted_mutual_info_score(labels_true, labels)
  print("Adjusted Mutual Information: %0.3f" %mutual_info_score)

  separation_index= (1- mutual_info_score)
  print("Separation Index: %0.3f" %separation_index)

  silhouette_score = metrics.silhouette_score(test_data, labels)
  print("Silhouette Coefficient: %0.3f" %silhouette_score)

  scores=[homogeneity_score, completeness_score, v_measure_score,
      rand_score, mutual_info_score,  silhouette_score]

  return scores

df = pd.read_csv("covid-data-2-3classes-cleaned.csv",index_col=0)
df = df.reset_index(drop=True)
df.head(5)

normalizaion_bit = 1
label_column_name = "file_label"
label_num = 3
label_names =  ['negative','positive','others']
legend_title = "COVID-19"

from scipy.stats import mode

def  resolve_mapping(y_pred, y):
  final_pred_labels=np.zeros_like(y)
  n_class=len(np.unique(y))
  for i in range(n_class):
    idx= (y_pred==i)
    final_pred_labels[idx]=mode(y[idx])[0]
  return final_pred_labels

def computeLearningHardIndex2(true_labels, predicted_labels):

  final_predicted_labels=resolve_mapping(predicted_labels, true_labels)
  mutual_info_score=metrics.adjusted_mutual_info_score(true_labels, final_predicted_labels)
  LHI=1-mutual_info_score
  return LHI

X = normalized_data=StandardScaler().fit_transform(df.drop(['file_label'],1))
y = df['file_label']
kmeans = KMeans(n_clusters=label_num, random_state=0).fit(X)

computeLearningHardIndex2(y, kmeans.labels_)