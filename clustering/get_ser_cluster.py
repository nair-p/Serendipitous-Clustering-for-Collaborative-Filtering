### This code is used to generate special "serendipitous" clusters in addition to the conventional clusters obtained ###
### using Spherical K-means || ###

import pickle
import numpy as np
from collections import Counter
import sys
from numpy import linalg as la
from math import sqrt
from distance_func import distance


# open the cluster centers obtained after conventional clustering carried out before this
with open('../data/centroids_half_400.pickle', 'rb') as handle:
	centers = pickle.load(handle)

# open the cluster label for each user obtained after conventional clustering carried out before this 
with open('../data/labels_half_400.pickle', 'rb') as handle:
	labels = pickle.load(handle)

with open('../data/user_rating_norm.pickle', 'rb') as handle:
	data = pickle.load(handle)

with open('../data/user_cast_profiles.pickle', 'rb') as handle:
	cast_profiles = pickle.load(handle)

with open('../data/user_genre_profiles.pickle', 'rb') as handle:
	genre_profiles = pickle.load(handle)


print("All necessary pickle files loaded")

data = np.array(data)

def findClosestPoint(ctr, point_ind):
# given a cluster centroid and the indices of points belonging to that cluster, returns the index of the closest point to the centroid
# mean -- also called the "concept" of that cluster
	points = list(data[point_ind])
	distances = distance(points, ctr)
	return np.argmin(distances)


def genreSimilarity(profile1, profile2):
# generates the cosine similarity between two normalized user genre profiles
	n1 = la.norm(profile1)
	n2 = la.norm(profile2)

	profile1 = profile1/n1
	profile2 = profile2/n2

	return np.dot(profile1, profile2.transpose())

def castSimilarity(profile1, profile2):
# generates the jaccard's similarity between two user cast profiles
	intersection = profile1 & profile2
	union = profile1 | profile2
	jacc_sim = len(intersection)/float(len(union))
	return jacc_sim

# set denoting the closest point (concept) of each cluster (center)
closestPoints = np.zeros(len(centers))

for c, ctrs in enumerate(centers):
# this loop finds the closest data point to each of the cluster centroids
	points = [i for i,l in enumerate(labels) if centers[l] == ctrs] 
	closestPoints[c] = findClosestPoint(ctrs, points)


# we take a weighted consideration of genre and cast preferences of users for assigning them to serendipitous clusters
# in the paper, 0.25 is alpha and 0.5 is beta
genre_weight = int(0.25*len(centers))
cast_weight = int(0.5*genre_weight)

serendipity_clusters = []

for uid in range(len(data)):

	genre_pref = []
	cast_pref = []

	label_ind = labels[uid]

	for c,p in enumerate(closestPoints):
		if c != label_ind:
			# take the users whose genre preferences are similar to the concept 
			genre_pref.append(tuple([genreSimilarity(genre_profiles[uid], genre_profiles[c]), c]))
			

	genre_pref = sorted(genre_pref)
	# take top least similar genres to introduce diversity
	genre_pref = genre_pref[0:genre_weight]

	for c,p in enumerate(genre_pref):
		# we take the users whose cast preferences are similar to the users for whom the genre preferences have been created
		cast_pref.append(tuple([castSimilarity(cast_profiles[uid], cast_profiles[c]), c]))

	cast_pref = sorted(cast_pref, reverse = True)
	# we take the top most similar casts to ensure relevance
	cast_pref = cast_pref[0:cast_weight]

	user_profile = []
	for c,p in cast_pref:
		# generating final user_profiles based on genre dissimilarity and cast similarity
		user_profile.append(p[1])


	serendipity_clusters.append(user_profile)

with open('../data/serendipity_clusters.pickle', 'wb') as handle:
	pickle.dump(serendipity_clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)










