### This code is used to give recommendations to users based on their predicted rating values. ###
### This code gives relevant and serendipitous recommendations based on the serendipitous clusters ### 

import pickle
import numpy as np
from collections import Counter
import sys
from numpy import linalg as la
from sklearn.metrics import mean_squared_error
from math import sqrt


with open('clustering/centroids_half_400.pickle', 'rb') as handle:
	centers = pickle.load(handle)

with open('clustering/labels_half_400.pickle', 'rb') as handle:
	labels = pickle.load(handle)

with open('data/user_ratings_norm.pickle', 'rb') as handle:
	data = pickle.load(handle)

with open('data/movies_map.pickle', 'rb') as handle:
	movie = pickle.load(handle)

with open('data/user_map.pickle', 'rb') as handle:
	user = pickle.load(handle)

with open('data/validation_users.pickle', 'rb') as handle:
	rec_users = pickle.load(handle)


def simi(u_ind, j):
	# returns the cosine similarity 
	return np.dot(data_norm[u_ind], data_norm[j])


print("All required pickle files loaded")

data = np.array(data)

rating_sum = np.sum(data, axis=1)

denom = np.count_nonzero(data, axis = 1)
rms = 0.0

# ground truth rating given to movies
mov_gt_rating = []

# ratings predicted for those same movies by our algorithm
predicted_rating = []
flag = 0 # flag is to check when to stop recommending movies

for key, value in rec_users.items():
	if(flag == 1):
		break
	num_mov = len(value)
	
	u_ind = user[key]
	u_avg = float(rating_sum[u_ind]/denom[u_ind])
	
	cluster_num = labels[u_ind]
	# obtaining cluster indices for each user 
	indices = [i for i, x in enumerate(labels) if x == cluster_num]
	
	for i in range(num_mov):

		if(flag == 1):
			break
		if(value[i][0] not in movie.keys()):
			continue	

		mov_ind = movie[value[i][0]] 
		
		sum_simi = 0.
		sum_net = 0.
		
		for j in indices:
			# following Pearson's predictive function for Collaborative Filtering 
			neighbour_rating = data[j][mov_ind]

			if(j==cluster_num or neighbour_rating == 0.0):
				continue
			
			neighbour_avg = float(rating_sum[j]/denom[j])
			
			simis = simi(u_ind, j)
			
			sum_net +=simis*(neighbour_rating - neighbour_avg)
			sum_simi+=simis
		
		if(sum_simi!=0):
			mov_gt_rating.append(value[i][1]) 
			predicted_rating.append(u_avg + sum_net/sum_simi)
			# we make predictions for 1000 movies
			if(len(predicted_rating)>999):
                flag = 1
                break


print("Number of gt ratings: %d" % len(mov_gt_rating))
print("Number of predictions: %d" % len(predicted_rating))

# calculating RMSE and MAE for our recommendations
rms = sqrt(mean_squared_error(mov_gt_rating, predicted_rating))
print("RMS error: %f" % rms)

mae = np.sum(np.absolute(np.array(mov_gt_rating) - np.array(predicted_rating)))/len(predicted_rating)
print("MA error: %f" % mae)
