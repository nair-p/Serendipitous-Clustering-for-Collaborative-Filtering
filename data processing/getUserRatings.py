### This code is used to generate the user rating matrix from the Serendipity 2018 dataset ###

import csv
import numpy as np
import pickle
from numpy import linalg as la

# Specify the number of users and number of movies you wish to take into consideration
#num_users = 104661
#num_movies = 49151
num_movies = 5
num_users = 2

# user_rating is the matrix containing user rating information
# Each row in the matrix corresponds to some user
# Each column in the matrix corresponds to an item (here - movie)
# Each cell (matrix[i][j]) represents the rating given by the ith user to the jth item
user_rating = [ [ 0. for i in range(num_movies) ] for j in range(num_users) ]


users = {}
movies = {}

ind1 = 0 # to keep track of number of users considered
ind2 = 0 # to keep track of number of movies considered

# firstline is to keep track of the first line in the datafile as its format is different from the remaining file
firstline = True
count = 0

#with open("./serendipity/training.csv") as csvfile:
with open("trial.csv") as csvfile:
	# read data files
	filereader = csv.reader(csvfile, delimiter='\n', quotechar='|')
	
	for line in filereader:
		# this code is specific to the format in which the Serendipity 2018 Dataset is presented
		count += 1
		
		if firstline:
			firstline = False
			continue
		
		line = line[0]
		row = line.split(',')
		
		userID = row[0].encode('utf-8')
		movieID = row[1].encode('utf-8')
		rating = row[2].encode('utf-8')
		
		if ind1 >= num_users and userID not in users.keys():
			# if the user does not fall within our set of users (from num_users)
			continue
		if ind2 >= num_movies and movieID not in movies.keys():
			# if the movie does not fall within our set of movies (from num_movies)
			continue

		if userID not in users.keys():
			# adding a newly encountered user to our database
			users[userID] = ind1
			ind1 += 1

		if movieID not in movies.keys():
			# adding a newly encountered movie to our database
			movies[movieID] = ind2
			ind2 += 1

		# updating our user rating matrix
		user_rating[users[userID]][movies[movieID]] = float(rating)
	
print("User rating matrix created")

# Normalise the user-movie rating information
norms = la.norm(user_rating, axis = 1)
user_rating_norm = user_rating / norms[:, None]

# Storing the necessary information in pickle files
with open('../data/user_map.pickle', 'wb') as handle:
	pickle.dump(users, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/movies_map.pickle', 'wb') as handle:
	pickle.dump(movies, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/user_rating_full.pickle', 'wb') as handle:
	pickle.dump(user_rating, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/user_rating_norm.pickle', 'wb') as handle:
	pickle.dump(user_rating_norm, handle, protocol=pickle.HIGHEST_PROTOCOL)

