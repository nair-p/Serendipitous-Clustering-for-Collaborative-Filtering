### This code is used to run the Lloyd's iterations of Spherical K-means clustering after any seeding technique ###

import numpy as np
from distance_func import distance

def KMeans(data, k, centroids, max_iter = 1000): 
    
    """ Apply the KMeans clustering algorithm
    
    Parameters:
      data                        ndarrays data 
      k                           number of cluster
      centroids                   initial centroids
    
    Returns:
      "Iteration before Coverge"  time used to converge
      "Centroids"                 the final centroids finded by KMeans    
      "Labels"                    the cluster of each data   
    """
    
    n = data.shape[0] 
    iterations = 0
    
    while iterations < max_iter:        
        dist = distance(data,centroids)
        
        ## give cluster label to each point 
        cluster_label = np.argmin(dist, axis=1)
        
        ## calculate new centroids
        newCentroids = np.zeros(centroids.shape)
        #print cluster_label
        for j in range(0, k):
            if sum(cluster_label == j) == 0:
                newCentroids[j] = centroids[j]
            else:
                newCentroids[j] = np.mean(data[cluster_label == j, :], axis=0)
                newCentroids[j] = np.divide(newCentroids[j], float(np.sqrt(np.sum(np.square(newCentroids[j])))))
        
        ## Check if it has converged
        if np.array_equal(centroids, newCentroids):
            print("Converge! after:",iterations,"iterations")
            break 
            
        centroids = newCentroids
        iterations += 1
        
    return({"Iteration before Coverge": iterations, 
            "Centroids": centroids, 
            "Labels": cluster_label})
