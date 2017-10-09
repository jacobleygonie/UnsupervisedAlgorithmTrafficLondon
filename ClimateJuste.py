#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:26:06 2017

@author: clairelasserre
"""
import xlrd
from time import time

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets


pull=[]
wb1 = xlrd.open_workbook('climate.xls')
sh1 = wb1.sheet_by_index(0)
X=[]    
for a in range(2,sh1.nrows):
    l=[]
    l.append(float(sh1.cell(a,6).value) )
    l.append(float(sh1.cell(a,19).value) )
    X.append(l)
#si on veut travailler sur un plus petit jeu de données 
X2 = []
for i in range (len(X)):
    if (i%4==0):
        X2.append(X[i])   
    
"""ETAPE 1 : volonté de travailler avec un 2d features, sensitivity - ability to prepare"""
"""COMPARER LES DIFFERENTES METHODES AGGLOMERATIVES"""
    
#----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(labels[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()

#----------------------------------------------------------------------

# 2D embedding of the digits dataset
def MethodesAgglomerativeClustering(X):
    print("Computing embedding")
    X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
    print("Done.")
    
    from sklearn.cluster import AgglomerativeClustering
    
    for linkage in ('ward', 'average', 'complete'):
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
        t0 = time()
        clustering.fit(X_red)
        print("%s : %.2fs" % (linkage, time() - t0))
    
        plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)
    
    
    plt.show()