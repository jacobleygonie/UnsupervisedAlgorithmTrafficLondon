"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from matplotlib import style
style.use("ggplot")




import math
from sklearn import linear_model
from sklearn import datasets





"""Nearest neighbor and the curse of dimensionality"""

data=[]
wb1 = xlrd.open_workbook('Ex16sat2.xlsx')
sh1 = wb1.sheet_by_index(0)
datanames=[]
target=[]

stationnames=[]

for a in range(4,sh1.nrows-1):
    stationnames.append(sh1.cell(a,1).value) 


for b in range (4,sh1.ncols-3):
    datanames.append(sh1.cell(2,b).value)

for a in range(4,sh1.nrows-1):
    ligne=[]
    for b in range(5,sh1.ncols-3) : 
        ligne.append(sh1.cell(a,b).value)
    data.append(ligne) 

for a in range(4,sh1.nrows-1):
    f = sh1.cell(a,2).value
    target.append(math.floor(f))

data_X = data
data_y = target

np.random.seed(0)
indices = np.random.permutation(len(data_X))
data_X_train = data_X[:-10]
data_y_train = data_y[:-10]
data_X_test = data_X[-10:]
data_y_test = data_y[-10:]
# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(data_X_train, data_y_train) 



res = knn.predict(data_X_test)

def distance (y1,y2):
    n1 = len(y1)
    n2 = len(y2)
    if (n1!=n2):
        print("vous ne pouvez pas comparez 2sets de taille différente")
    else :
        error=0
        for i in range (n1):
            if (y1[i]!=y2[i]):
                error =error+1
    return error
"""pour l'améliorer on peut prendre la moyenne des 10 premiers termes, les 10 autres..."""

y=[1,2,3,4]
from statistics import mean
def reductiondata(y):
    n = len(y)
    q = math.floor(n/10) 
    r = n-q*10
    m=[]
    if(q!=0):
            for i in range (10):
                m.append(mean(y[i*q:(i+1)*q]))
                m.append(mean(y[10*q:10*q+r]))
    return m

"""LINEAR regression"""

regr = linear_model.LogisticRegression(penalty='l2',C=1e2)
regr.fit(data_X_train, data_y_train)

regr1 = linear_model.LinearRegression()
regr1.fit(data_X_train, data_y_train)


# The mean square error
print(np.mean((regr.predict(data_X_test)-data_y_test)**2))


# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
sc = regr.score(data_X_test, data_y_test) # on trouve sc de -6