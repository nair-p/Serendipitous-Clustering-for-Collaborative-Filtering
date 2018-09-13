### This code is part of the SPKM|| code and is used to choose the k initial centers after seeding ###

import numpy as np
from distance_func import distance

def w_cost(dist, weight):
    """ Calculate the cost of data with respect to the current centroids
    Parameters:
       dist     distance matrix between data and current centroids
       weight   weights assigned to each of the centroids

    Returns:    the normalized constant in the distribution 
    """
    
    return np.sum(weight*np.min(dist,axis=1))

def w_distribution(dist,cost,weight):
    """ Calculate the distribution to sample new centers
    Parameters:
       dist       distance matrix between data and current centroids
       cost       the cost of data with respect to the current centroids
       weight     weights assigned to each of the centroids
    Returns:      distribution 
    """
    return weight*np.min(dist, axis=1)/cost

def w_sample_new(distribution,l):
    """ Sample new centers
    
    Parameters:
       distribution n*1
       l            the number of new centers to sample
    Returns:        indices of new centers                          
    """
    return np.random.choice(range(len(distribution)),l,p=distribution, replace=False)

def wkmeanspp(data, cent_pos, k, w, one_ind):    
    """ Apply the KMeans++ clustering algorithm to get the initial centroids   
    Parameters: 
      data                        ndarrays data 
      cent_pos                    indices of the selected centroids
      k                           number of cluster
      w                           weights assigned to centroids
      one_ind                     index of the first randomly sampled centroid
    
    Returns:
      actual_cent                 the complete initial centroids by SPKM++
      centroids                   the indices of the initial centroids
      
    """
    
    #Initialize the first centroid
    centroids = data[one_ind,:]
    cent_pos_wk = np.array([one_ind])
    actual_cent = np.array([cent_pos[one_ind]])
    #print(len(data))
    while centroids.shape[0] < k :
        
        #Get the distance between data and centroids
        dist = distance(data, centroids, cent_pos_wk)
        
        #print(dist.shape)
        #Calculate the cost of data with respect to the centroids
        norm_const = w_cost(dist, w)

        
        #Calculate the distribution for sampling a new center
        p = w_distribution(dist,norm_const,w)
        #print(len(p))

        #Sample the new center and append it to the original ones
        pos = w_sample_new(p,1)
        #print(pos)
        #print("###")
        
        
        cent_pos_wk = np.append(cent_pos_wk, pos)
        actual_cent = np.append(actual_cent, cent_pos[pos])
        centroids = np.r_[centroids, data[pos,:]]
    
    return centroids, actual_cent
