##### Kmeans Clustering with Euclidean distance for k=1:10#####
##### Precision, Recall and F-score for each of the clusters#####
##### Marta Pinol #####

import numpy as np
import matplotlib.pyplot as plt

# Euclidian distance
def dist(a, b):
    return np.linalg.norm(a - b)

# Number of Permutations(n,r)
def nPermutations(n,r):
    a = 1
    b = 1
    for i in range(1,r+1):
        a = a *  i
        b = b * (n-i+1)
    return b/a

# Kmeans Clustering 
def KmeansClustering(data, k, MaxIterations, seed = 1997):
    
    # Initialisation
    np.random.seed(seed)
    centers = np.random.choice(range(len(data)), k, replace=False) # k random points as centers
    centroids = [data[i] for i in centers]
    centroids0 = np.zeros(np.shape(centroids)) # centroids0 have coordinates 0
    error = [dist(centroids0[j], centroids[j]) for j in range(k)] # euclidian distance between centroids0 and initial centroids
    iter = 0;    

    # Iterations until the error becomes zero or it reaches the max number of iterations
    while (np.sum(error)!=0 and iter <= MaxIterations):
        iter = iter + 1
        clusters = []
		
		# Compute distance between each of the points with each of the centroids
        for i in range(len(data)):
            x = data[i]
            distances = []
            for j in range(k):
                distances = distances + [dist(x, centroids[j])]
			
			# each point is assigned in the cluster of the nearest centroid
            clusters = clusters + [np.argmin(distances)]

        # Finding the new centroids by taking the average value
        new_centroids = []
        error = []
        for j in range(k):
            points = [data[i] for i in range(len(data)) if clusters[i] == j]
            new_centroids = new_centroids + [np.mean(points, axis=0)]
            error = error + [dist(new_centroids[j], centroids[j])]

        centroids = new_centroids

    return clusters, iter

# Precision, Recall and Fscore 
def PrecisionRecallFscore(k,clusters,y):

	# Count how many points in each real class
    classes = dict((i, list(y).count(i)) for i in y)
    count_classes = list(classes.values())
    
	# Compute Total Trues
    Total_Trues = 0
    for i in range(len(count_classes)):
        Total_Trues = Total_Trues + nPermutations(count_classes[i],2)

	# Compute Total Positives and True Positives per cluster
    total_positives = []
    true_positives = []
    for j in range(k):
        instances = [y[i] for i in range(len(y)) if clusters[i] == j]
        total_positives = total_positives + [nPermutations(len(instances),2)]
        count = dict((i, instances.count(i)) for i in instances)
        count_values = count.values()
        max_value = max(count_values)
        true_positives = true_positives + [nPermutations(max_value,2)]
    
	# Total Positives and True Positives in total
    sum_total_positives = sum(total_positives)
    sum_true_positives = sum(true_positives)

	# Compute Precision, Recall and Fscore 
    Precision = sum_true_positives/sum_total_positives
    Recall = sum_true_positives/Total_Trues
    Fscore = 2*(Precision*Recall)/(Precision+Recall)

    return Precision,Recall,Fscore
