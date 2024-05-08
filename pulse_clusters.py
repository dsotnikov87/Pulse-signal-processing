# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:23:39 2024

@author: sotnik_d
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 6, 4

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#%% Selection of the best cluster number by Silhouette method

# https://medium.com/inst414-data-science-tech/clustering-data-science-jobs-e9466e6d7c31
# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
  sse = []
  for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse

def single_kmeans(x, kmax):
    
    sil = []
    
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(x)
        labels = kmeans.labels_
        sil.append(silhouette_score(x, labels, metric = 'euclidean'))
        
    sil = np.array(sil)

    return sil

# Gaussian function could be used in case of deviation the number
# of centroids of k-means
def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

#%% 

def pulse_clusters(t, I):
    
    x = np.concatenate((t.reshape(-1, 1), I.reshape(-1, 1) ), axis = 1)
    
    # number of clusters by current value

    kmax = 5

    sil = single_kmeans(x[:, 1].reshape(-1, 1), kmax)

    kpeak = int(np.arange(2, kmax+1)[np.where(sil == np.max(sil))][0])

    # number of clusters by time

    clustering_kmeans = KMeans(n_clusters = kpeak)
    clusters_I = clustering_kmeans.fit_predict(x[:, 1].reshape(-1, 1))

    kmax = 20

    ktot = 0

    borders = np.zeros((1, 2))

    for i in np.unique(clusters_I):
        
        tx = x[:, 0][clusters_I == i]
        Ix = x[:, 1][clusters_I == i]
        
        sil = single_kmeans(tx.reshape(-1, 1), kmax)

        kpeak = int(np.arange(2, kmax+1)[np.where(sil == np.max(sil))][0])
        
        ktot += kpeak

        clustering_kmeans = KMeans(n_clusters = kpeak)
        clusters = clustering_kmeans.fit_predict(tx.reshape(-1, 1))
        
        for j in np.unique(clusters):
            
            borders = np.append(borders, np.array([np.min(tx[clusters == j]), np.max(tx[clusters == j])]).reshape(1, -1), axis = 0)

    borders = borders[1:]

    borders = borders[borders[:, 0].argsort()]
    
    return borders