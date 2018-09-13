### This code is used to cluster any set of points using the scalable K-Means parallel algorithm ###
### The data points need to be in a n*d matrix form -- each row corresponds to one data point of d-dimensions ###

import numpy as np
from distance_func import distance, distance_kmeans
from kmeanspp_func import cost_s, distribution, sample_new
from wkpp_func import wkmeanspp
from kmeans_func import KMeans

def clusterCostseed(data,predict, cent_pos_wk):
    # clustering cost right after seeding initial k-centers
    dist = distance(data,predict, cent_pos_wk)
    return cost_s(dist, len(data))/(10**2)

def clusterCost(data,predict):
    # clustering cost after clustering
    dist = distance_kmeans(data,predict)
    return cost_s(dist, len(data))/(10**2)


def get_weight(dist,centroids,cent_pos):
    # to obtain the weights of each centroid to select final k clusters
    # the weight of a centroid is the number of data points assigned to that cluster
    s = 0.
    min_dist = np.zeros(dist.shape)
    min_dist[range(dist.shape[0]), np.argmin(dist, axis=1)] = 1
    min_dist[cent_pos] = 0
    count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(centroids.shape[0])])
    s += np.sum(count)
    
    return count, s

def ScalableKMeansPlusPlus(data, k, l,r):
    
    cent_pos = np.random.choice(range(data.shape[0]),1)
    centroids = data[cent_pos, :]
    
    for i in range(1,r+1):
        
        #Get the distance between data and centroids
        dist = distance(data, centroids, cent_pos)
        #Calculate the cost of data with respect to the centroids
        norm_const = cost_s(dist, len(data)) 
        #Calculate the distribution for sampling l new centers
        p = distribution(dist,norm_const)
        #Sample the l new centers and append them to the original ones
        pos = sample_new(p,l)

        cent_pos = np.append(cent_pos, pos)
        centroids = np.r_[centroids, data[pos]]
        
    dist = distance(data, centroids, cent_pos)
    w, s = get_weight(dist, centroids, cent_pos)
        
    weights = w/s

    centroid_one_ind = np.random.choice(len(weights), 1, p = weights)
    # employing weighted Spherical K-Means ++ to obtain final k cluster centers
    centroids_ini_spkm_para, cent_pos_wk = wkmeanspp(centroids, cent_pos, k, w, centroid_one_ind)
    
    return centroids_ini_spkm_para