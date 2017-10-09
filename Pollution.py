#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:34:43 2017

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
from sklearn.cluster import KMeans,MiniBatchKMeans,AffinityPropagation,MeanShift, estimate_bandwidth
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
import time
from sklearn.metrics.pairwise import pairwise_distances_argmin


pull=[]
wb1 = xlrd.open_workbook('LAEI2010_GridSplits_Oil_Areas_all_years.xls')
sh1 = wb1.sheet_by_index(0)



"""ETAPE 1 : on fait u k means avec comme features uniqument le taux de CO2 annuel 
motif : ici comme 1 feature, 1 graphe aurait suffi mais le but est de prédire pour obtenir un graphe du type voronoid (ie continu)"""

"""
X=[]
for a in range(1,sh1.nrows-1):
        X.append(sh1.cell(a,7).value)   
Y=[]
for a in range(1,sh1.nrows-1):
        Y.append(sh1.cell(a,8).value)  
Area=[]
for a in range(1,sh1.nrows-1):
        Area.append(sh1.cell(a,8).value)  
CO2=[]       
for a in range(1,sh1.nrows-1):
        CO2.append([float(sh1.cell(a,14).value)] )
        
n=len(CO2)



def LondonMapCO2Kmeans():

    kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
    kmeans.fit(CO2)
    labels=kmeans.labels_
    
    data0 =[]
    data1=[]
    data2=[]
    data3=[]
    X0=[]
    Y0=[]
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    X3=[]
    Y3=[]
    X4=[]
    Y4=[]
    for i in range (len(CO2)):
        if(labels[i]==0):
            data0.append(CO2[i])
            X0.append(X[i])
            Y0.append(Y[i])
        if(labels[i]==1):
            data1.append(CO2[i])
            X1.append(X[i])
            Y1.append(Y[i])
        if(labels[i]==2):
            data2.append(CO2[i])
            X2.append(X[i])
            Y2.append(Y[i])
        if(labels[i]==3):
            data3.append(CO2[i])
            X3.append(X[i])
            Y3.append(Y[i])
        if(labels[i]==4):
            data3.append(CO2[i])
            X4.append(X[i])
            Y4.append(Y[i])
            
    x_min, x_max = min(X)-1,max(X)+1
    y_min,y_max =  min(Y)-1,max(Y)+1
    
    plt.figure(1)
    plt.clf()
    centroids = kmeans.cluster_centers_
    
    plt.plot(X0,Y0,'o',marker='.',color='g',label=centroids[0])
    plt.plot(X1,Y1,'o',marker='.',color='b',label=centroids[1])
    plt.plot(X2,Y2,'o',marker='.',color='c',label=centroids[2])
    plt.plot(X3,Y3,'o',marker='.',color='r',label=centroids[3])
    plt.plot(X4,Y4,'o',marker='.',color='w',label=centroids[4])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.title("CO2 dans Londres")
    #plt.legend()
    plt.show()
    print("La zone verte correspond à une zone de CO2 moyenne " ,  centroids[0][0] , 'tonnes par an')
    print("La zone bleue correspond à une zone de CO2 moyenne " ,  centroids[1][0] , 'tonnes par an')
    print("La zone turquoise correspond à une zone de CO2 moyenne " ,  centroids[2][0] , 'tonnes par an')
    print("La zone rouge correspond à une zone de CO2 moyenne " ,  centroids[3][0] , 'tonnes par an')
    print("La zone blanche correspond à une zone de CO2 moyenne " ,  centroids[4][0] , 'tonnes par an')
    
    
"""



"""PARTIE 2 : prendre en compte différentes données de pollution"""

"""
X=[]
for a in range(1,sh1.nrows-1):
        X.append(sh1.cell(a,7).value)   
Y=[]
for a in range(1,sh1.nrows-1):
        Y.append(sh1.cell(a,8).value)  
data=[]       
for a in range(1,sh1.nrows-1):
    l=[]
    for j in range (14,sh1.ncols):
        l.append(float(sh1.cell(a,j).value) )
    data.append(l)

    
def LondonMapAllPollutionKmeans():  
    kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
    kmeans.fit(data)
    labels=kmeans.labels_
    
    data0 =[]
    data1=[]
    data2=[]
    data3=[]
    X0=[]
    Y0=[]
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    X3=[]
    Y3=[]
    X4=[]
    Y4=[]
    for i in range (len(data)):
        if(labels[i]==0):
            data0.append(data[i])
            X0.append(X[i])
            Y0.append(Y[i])
        if(labels[i]==1):
            data1.append(data[i])
            X1.append(X[i])
            Y1.append(Y[i])
        if(labels[i]==2):
            data2.append(data[i])
            X2.append(X[i])
            Y2.append(Y[i])
        if(labels[i]==3):
            data3.append(data[i])
            X3.append(X[i])
            Y3.append(Y[i])
        if(labels[i]==4):
            data3.append(data[i])
            X4.append(X[i])
            Y4.append(Y[i])
            
    x_min, x_max = min(X)-1,max(X)+1
    y_min,y_max =  min(Y)-1,max(Y)+1
    
    plt.figure(1)
    plt.clf()
    centroids = kmeans.cluster_centers_
    
    plt.plot(X0,Y0,'o',marker='.',color='g',label=centroids[0])
    plt.plot(X1,Y1,'o',marker='.',color='b',label=centroids[1])
    plt.plot(X2,Y2,'o',marker='.',color='c',label=centroids[2])
    plt.plot(X3,Y3,'o',marker='.',color='r',label=centroids[3])
    plt.plot(X4,Y4,'o',marker='.',color='w',label=centroids[4])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.title("tous types de polluants Londres dans Londres")
    
    plt.show()
"""
"""PARTIE 3 : comparaison des méthodes en terme de temps (difficile efficacité car on a pas les labels_true)"""


"""
X=[]
for a in range(1,sh1.nrows-1):
        X.append(sh1.cell(a,7).value)   
Y=[]
for a in range(1,sh1.nrows-1):
        Y.append(sh1.cell(a,8).value)  
data=[]       
for a in range(1,sh1.nrows-1):
    l=[]
    for j in range (14,sh1.ncols):
        l.append(float(sh1.cell(a,j).value) )
    data.append(l)
    
#datareduce correspond à data mais avec uniquement 2 features, CO2 et un autre, on ne l'utilise que pour montrer 
#..l'absurdité de la méthode affinityPropragation
datareduce =[]
for a in range(1,sh1.nrows-1):
        l=[]
        l.append(float(sh1.cell(a,14).value) )
        l.append(float(sh1.cell(a,20).value) )
        datareduce.append(l)


        
def ClusteriserData(methode,nclusters,data):
    if (methode==1) : 
        kmeans = KMeans(init='k-means++',algorithm='auto', n_clusters=nclusters, n_init=10)
        t0 = time.time()
        kmeans.fit(data)
        labels=kmeans.labels_
        centroids = kmeans.cluster_centers_
        delta_t = time.time() - t0
    
    if (methode==2):
        batch_size = 45
        mbk = MiniBatchKMeans(init='k-means++', algorithm='auto',n_clusters=nclusters, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
        t0 = time.time()
        mbk.fit(data)
        labels=mbk.labels_
        centroids = mbk.cluster_centers_
        delta_t = time.time() - t0
    
        
    if (methode==3): #donne une idée du nombre de clusters idéals : 38 ! #expliquer que trop de valeur donc il faut se restreindre pour ploter
        t0 = time.time()
        af = AffinityPropagation(convergence_iter =5).fit(datareduce[:100]) 
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        centroids = af.cluster_centers_
        nclusters=(len(cluster_centers_indices))
        delta_t = time.time() - t0
    
    
    if (methode==4):
        # The following bandwidth can be automatically detected using
        data = np.array(data)
        t0 = time.time()
        bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
        
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        centroids = cluster_centers 
        labels_unique = np.unique(labels)
        nclusters = len(labels_unique)
        delta_t = time.time() - t0
        print("number of estimated clusters : %d" % nclusters)

    return [labels, centroids, delta_t,nclusters]

def PerfomanceKmeansBatch(data):
    TK = []
    TM=[]
    N=[]
    batch_size = 45
    for i in range (1,20):
        N.append(i)
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
        t0 = time.time()
        kmeans.fit(data)
        delta_t = time.time() - t0
        TK.append(delta_t)
        minkmeans = MiniBatchKMeans(init='k-means++',n_clusters=i, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
        t0 = time.time()
        minkmeans.fit(data)
        delta_t = time.time() - t0
        TM.append(delta_t)
    plt.plot(N,TM, label = 'kmeans')
    plt.plot(N,TK,label = 'MiniBatchKMeans')
    plt.legend()
    plt.xlabel("n_clusters")
    plt.ylabel("Time (ms)")
    plt.title("Comparision Kmeans & MiniBatchKMeans")
    plt.show()
    
def ResultKmeansMinbatch(data,n_clusters):
    data=np.array(data)
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means.fit(data)
    batch_size = 45
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
    
    mbk.fit(data)
    
    
    ##############################################################################
    # Plot result
    
    
    
    # We want to have the same colors for the same cluster from the
    # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
    # closest one.
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(data, k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(data, mbk_means_cluster_centers)
    order = pairwise_distances_argmin(k_means_cluster_centers,
                                      mbk_means_cluster_centers)
    Xsimilar = []
    Ysimilar = []
    Xdifferent=[]
    Ydifferent=[]
    for i in range (len(k_means_labels)):
       
        if (k_means_labels[i]==mbk_means_labels[i]):
            Xsimilar.append(X[i])
            Ysimilar.append(Y[i])
        else : 
            Xdifferent.append(X[i])
            Ydifferent.append(Y[i])
    
    x_min, x_max = min(X)-1,max(X)+1
    y_min,y_max =  min(Y)-1,max(Y)+1
    plt.plot(Xsimilar,Ysimilar,'o',color='w', marker='.')
    plt.plot(Xdifferent,Ydifferent,'o',color='r', marker='.')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.title("points de divergence entre la méthode kmeans et MiniBatch")
    
    plt.show()

def GrapheLondon (methode,nclusters,X,Y,data):  
    [labels, centroids, delta_t,nclusters] = ClusteriserData(methode,nclusters,data)
    datatot=[] #ligne 0 correspond à label =0, ligne1 -> label=1
    Xtot=[] #dememe pour X, ligne 0 =valeur de X pour les élements qui ont pour label 0
    Ytot=[]
    for j in range (nclusters):
        datatot.append([])
        Xtot.append([])
        Ytot.append([])
    for i in range (len(labels)):
        for j in range (nclusters):
            if(labels[i]==j):
                datatot[j].append(data[i])
                Xtot[j].append(X[i])
                Ytot[j].append(Y[i])
        
    x_min, x_max = min(X)-1,max(X)+1
    y_min,y_max =  min(Y)-1,max(Y)+1
    from itertools import cycle
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(nclusters), colors):
        plt.plot(Xtot[k],Ytot[k],'o',marker='.',color=col,label=centroids[k])
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.title("Répartition des polluants dans Londres, kmeans ")
    plt.show()
    
            
def methodeAffinityPropragation(): #ronction qui réalise l'apprentissage et print, le but est de montrer que pas adapter car trop de linéarité entre les features 
  #comparer avec le beau graphe  http://scikit-learn.org/stable/modules/clustering.html#k-means en 2.3.3

    datareduce =[]
    for a in range(1,sh1.nrows-1):
        l=[]
        l.append(float(sh1.cell(a,14).value) )
        l.append(float(sh1.cell(a,20).value) )
        datareduce.append(l)
    datareduce = datareduce[:100]
    t0 = time.time()
    af = AffinityPropagation(convergence_iter =5).fit(datareduce) 
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    centroids = af.cluster_centers_
    dt = time.time()-t0
    n_clusters_=(len(cluster_centers_indices))   
    
    
    import matplotlib.pyplot as plt
    from itertools import cycle
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        cluster_center = centroids[k]
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k')
        Feat0 =[]
        Feat1=[]
        for i in range (len(datareduce)):
            if (labels[i]==k):
                Feat0.append(datareduce[i][0])
                Feat1.append(datareduce[i][1])
        plt.plot(Feat0, Feat1, col + '.')
    plt.title('Affinity propragation on CO2-NO2 features')
    print()
    print('running time', dt)
    plt.xlabel("CO2")
    plt.ylabel("NO2 ")
    print()
    print('nombre de clusters estimés',n_clusters_)
    return af.get_params();


"""   



"""PARTIE 4 : le voronoi avec réduction des features pour arriver en dim2"""
"""X=[]
for a in range(1,sh1.nrows-1):
        X.append(sh1.cell(a,7).value)   
Y=[]
for a in range(1,sh1.nrows-1):
        Y.append(sh1.cell(a,8).value)  
data=[]       
for a in range(1,sh1.nrows-1):
    l=[]
    for j in range (14,sh1.ncols):
        l.append(float(sh1.cell(a,j).value) )
    data.append(l)
n_digits=5
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 10), np.arange(y_min, y_max, 0.2))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()"""