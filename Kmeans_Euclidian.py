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


# Read data

print("Enter the location of the data")
data_folder = input()
animals = []
with open(data_folder + "animals","r") as f:
    lines = [elem for elem in f.read().split('\n') if elem]
for line in lines:
    animals.append(line.split())

countries = []
with open(data_folder + "countries","r") as f:
    lines = [elem for elem in f.read().split('\n') if elem]
for line in lines:
    countries.append(line.split())

fruits = []
with open(data_folder + "fruits","r") as f:
    lines = [elem for elem in f.read().split('\n') if elem]
for line in lines:
    fruits.append(line.split())

veggies = []
with open(data_folder + "veggies","r") as f:
    lines = [elem for elem in f.read().split('\n') if elem]
for line in lines:
    veggies.append(line.split())

# Combine data
data = animals + countries + fruits + veggies

# Keep real classes
classes = np.repeat(["animals","countries","fruits","veggies"], [len(animals),len(countries),len(fruits),len(veggies)], axis=0)

# Keep only numerical variables for the Kmeans Clustering
X = [np.float_(data[i][1:]) for i in range(len(data))]

# Set max number of iterations of the Kmeans Clustering
maxIterations = 100

# Perform Kmeans Clustering and its Prescison Recall and F-score for k=1:10
clusters,iterations = KmeansClustering(X,1,maxIterations)
print("\nNumber of elements per cluster k = ", 1)
print(dict((i, clusters.count(i)) for i in clusters))
results = [PrecisionRecallFscore(1,clusters,classes)]
for k in range(2, 11):
    clusters,iterations = KmeansClustering(X,k,maxIterations)
    print("\nNumber of elements per cluster k = ", k)
    print(dict((i, clusters.count(i)) for i in clusters))
    results = np.append(results, [PrecisionRecallFscore(k,clusters,classes)], axis = 0)

# Printing results regarding the correctnes of the Kmeans Clustering
print("\nPrecision, Recall and F-score k=1...10")
print(results)
precisions = results[:,0]
recalls = results[:,1]
fscores = results[:,2]

# Plot results regarding the correctnes of the Kmeans Clustering
import matplotlib.pylab as pylab
params = {'legend.fontsize': 30,
         'axes.labelsize': 30,
         'axes.titlesize': 30,
         'xtick.labelsize': 30,
         'ytick.labelsize': 30}
pylab.rcParams.update(params)

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)

plt.plot(range(1, 11), precisions, label="Precision")
plt.plot(range(1, 11), recalls, label="Recall")
plt.plot(range(1, 11), fscores, label="F-score")

plt.xticks(np.arange(1, 11, 1))
plt.yticks(np.arange(0.2, 1.1, 0.1))

# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


plt.show()

