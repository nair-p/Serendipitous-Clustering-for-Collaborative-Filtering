### This file contains the code to find the cosine distance between the cluster centers and other data points ###

import numpy as np

def distance(data, centroids):
    """ Calculate the distance from each data point to each center
    Parameters:
       data   n*d
       center k*d
    
    Returns:
       distence n*k 
    """
    dist = np.dot(data, centroids.transpose())
    dist = 1.5 - dist

    return dist
