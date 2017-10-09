#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:15:21 2017

@author: clairelasserre
"""

import matplotlib.pyplot as plt
import xlrd
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd

pull=[]
wb1 = xlrd.open_workbook('Ex16sat.xltx')
sh1 = wb1.sheet_by_index(0)
datanames=[]
dataminutes= []
stationnames=[]


    

for b in range (4,sh1.ncols-3):
    datanames.append(sh1.cell(2,b).value)

s0=120
for i in range (96):
    if (s0==1440): 
        s0=0
    dataminutes.append(s0)
    s0=s0+15
    
    

for a in range(4,sh1.nrows-1):
        stationnames.append(sh1.cell(a,1).value)    
    #je veux que la ligne 0 de pull soit la data0
for b in range(4,sh1.ncols-3) : 
    ligne=[]
    for a in range(4,sh1.nrows-1):
        ligne.append(sh1.cell(a,b).value)
    pull.append(ligne)

    #ici pull est donc un tableau de ligne : ligne 0 = entree pour toutes les stations pour data 0, 
    #donc autant de lignes que de plages horaires et de colonnes que de stations
data = pull 

def bench_k_means(estimator, data):
    t0 = time()
    estimator.fit(data)
    print('durée pour la clusterisation : ')
    print(time() - t0)
    print("les labels sont ", estimator.labels_)
    


              
def GetCentroid(estimator):
    return estimator.cluster_centers_
    
def GetCentroidAverageCoor(estimator):
    CentroidAverage=[]
    centroids = GetCentroid(estimator)
    nstations = len(centroids[0])
    for i in range (len(centroids)):
        m=0
        for j in range (nstations):
            m = m+centroids[i,j]
        CentroidAverage.append(m/(nstations))
    return CentroidAverage


# Plot the Models Classifications
# Store the inputs as a Pandas Dataframe and set the column names
def DensityLongonUnderground(nclusters,init):
    X= dataminutes 
    Y=[]
    
    if (init==1):
        estimator = KMeans(init='k-means++', n_clusters=nclusters, n_init=10)
        bench_k_means(estimator, data=data)
    else:
        estimator = KMeans(init='random', n_clusters=nclusters, n_init=10)
        bench_k_means(estimator, data=data)
    labels =   estimator.labels_
    centroids = GetCentroidAverageCoor(estimator)
    for i in range(len(labels)):
        Y.append(centroids[labels[i]])
    n=len(X)
    Xprime = X[:n-8]
    Xprime = X [n-8:n]+Xprime
    Yprime = Y[:n-8]
    Yprime = Y [n-8:n]+Yprime
    
    
    plt.plot(Xprime,Yprime)
    #plt.axis([0, 750, 0, 300])
    plt.xlabel('temps en minutes après minuit')
    plt.ylabel('densité centroid associé')
    plt.show()
    


