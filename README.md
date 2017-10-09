# UnsupervisedAlgorithmTrafficLondon
Several clustering methods used on some datasets from LondonTube such as the pollutions rates in the city and the bike/tube traffic

Warning: the database is not provided here. In order to get it, you should either contact me at jacob.leygonie@polytechnique.edu or my colleague Claire Lasserre at claire.lasserre@polytechnique.edu

Different interesting parts in the projetRatp directory

I. ClimateChange directory

You may use ClimateJuste.py in a GUI.

If you run it, you may call the MethodesAgglomerativeClustering fonction with X as argument in order to compare agglomerative methods with different linkage criteria: ward, average, complete.
It returns the three cluster graphs where data is some areas of London, and features are the sensitivity to environnemental changes and the ability to prepare.

II. Pollution directory

You may run pollution.py:

Part 1
-if you call LondonMapCO2Kmeans(), you will see the cluster graphs done by a KMeans algorithm. Notice that we actually see a map of London with locations in different clusters corresponding to there CO2 pollution rates.

Part 2
-if you call LondonMapAllPollutionKmeans(), you will have a similar result; nervertheless, the data contains several pollutions criteria (NO2,CO2,… 14 in total).

Part 3
-We here compare principal methods of unsupervised Machine Learning. We keep pollution data as in Part 2. 

-The main function is ClusteriserData(methode, ncluster, data). You can try it with « data » as data argument and any integer for ncluster according to your taste. If you choose « 1 » for method, it will run kmeans, « 2 » for Minbatch means, « 3 » for Affinity Propagation, « 4 » for MeanShift. 

-PerformanceKmeansBatch(data) plots the time it requires to predict data with kmeans and MinBatch kmeans methods in order to compare those two methods.

-ResultKmeansMinbatch(data,n_clusters) plots London map with points clustered differently by kmeans and MinBatch kmeans methods.

-GraphLondon(method,ncluster,X,Y,data) (cluster, data and method are to be chosen as above). You can just fill X and Y arguments with « X » and « Y », which correspond to the coordinates vectors of areas in London. 

-methodAffinityPropagation(): You can call it in order to check that this method is unappropriate. For 100 datas, in finds 37 clusters. PS: it uses dataReduce, 2D features (NO2-CO2 rates) instead of data.



III. London Bike directory

You may run bike.py

It is made to analyze and plot a large amount of datas about the amount of available bikes in different areas in London. It will by default use the file « 31JourneyDataExtract09Nov2016-15Nov2016.csv ». Line 89, you can add in the frames list some other files (mentioned above). We advice that you choose one to execute the following functions efficiently for a first time. N.B: in order not to overwhelm the moodle capacity, we only gave the files from number 28 to 32.

By default, if you run it in your python terminal, it will plot an evolving 3D graph. The X and Y coordinates are for locations of London Areas. The Z coordinate indicates the relative number of bikes. The graph move with the time of day evolving.

If you comment this part (line 517 to 548) and uncomment the following (line 550 to 604), you will have a 2D evolving plot. PS: you will have to wait a few seconds during the plot in order to see how negative becomes the number of bikes in some areas.

Finally, the make_linearisation(i,h,C) function may be run in order to try different SVM regression (linear, rbf, linearsvc, polynomial). The data is some periods of the day for a specific area, and the feature the amount of bike. You may choose any « i  » you want to choose a station, put 0.2 as h and 1 for C, which are parameters of SVM methods. 
We actually find out the results to be poor.

IV LondonTube directory

Opening de densite directory, you can use projet1.py 
It focuses on the LondonTube. Data are the subway stations and the features are the number of passing runs by laps of 15 min time (so 96 features)
Once it is run, the essential function calling all the other is DensityLondonUnderground(nclusters,init). You may choose any integer you want for nclusters, and « 1 » if you want kmeans++ initialization or « 2 » if you need a random initialization.

Warning

It is possible that your version of xcel or csv is not compatible with the Python librairies used in the code. We found those kind of problems while transferring algorithm to the members of the group.



