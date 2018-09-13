### This is a helper function used to cluster the users using the SPKM || clustering algorithm ###

import numpy as np
from math import sqrt 
from kmeans_func import KMeans
from kmeanspp_func import KMeansPlusPlus, cost
from scalablekmeanspp_func import ScalableKMeansPlusPlus
import matplotlib.pyplot as plt
from distance_func import distance
import pickle
import time

def clusterCost(data,predict):
	 # returns cost of clustering 
     dist = distance(data,predict["Centroids"])
     return cost(dist, len(data))/(10**1)

## Simulate data
k = 20
n = 5000
d = 41073
l = k

with open('../data/user_rating_norm.pickle', 'rb') as handle:
	data = pickle.load(handle)

data = np.array(data)


tot1 = 0
tot2 = 0	
best_cost1 = 0
best_cost2 = 0

num_iter = 3

print("PARALLEL SEED")

# repeat the experiment for num_iter times
for i in range(num_iter):
	s2 = time.time()
	# obtain initial centroids using the SPKM || algorithm
	centroids_initial = ScalableKMeansPlusPlus(data, k, l)
	t2 = time.time() - s2
	print("Seeding done")
	
	# obtain final clusters after Lloyd's iterations
	output_spp_para = KMeans(data, k, centroids_initial)
	
	tot2 += t2
	cst = clusterCost(data, output_spp_para)
	
	best_cost2 += cst



centroids2 =output_spp_para["Centroids"]
labels2 = output_spp_para["Labels"]

#print(centroids2)
#print(labels2)


'''
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, k)]
print colors

#print data[[3,4,5], :][:,0]
print data[[0,1,2],:][:,0]
#print data[labels1==1, :][:,0]
for i,color in enumerate(colors,start =1):
    plt.scatter(data[labels1==i, :][:,0], data[labels1==i, :][:,1], color=color)
    plt.axis([0, 110, 0, 110])

for j in range(k):
    plt.scatter(centroids1[j,0],centroids1[j,1],color = 'k',marker='x')

plt.show()
'''


print("")
print("FINAL RESULTS")
print("Clustering Cost:")
print("SPKM ||:", best_cost2/num_iter) # Scalable KMeans++

print("\nSeeding Time:")
print("SPKM ||:", tot2/num_iter) # Scalable KMeans++




