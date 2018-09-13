### This code is used to generate the movie features from Serendipity 2018 dataset ###

import csv
import numpy as np
import pickle
import collections
import datetime

# specify the number of movies you wish to generate features for
num_movies = 49151

movie_cast = [None for i in range(num_movies)] # stores the cast information of the movies
movie_date = [[0] for j in range(num_movies)] # stores the information about the date of release of the movies 

dates = {}
#directors = {}
#top_cast = {}

# these movie genres have been retrieved from the MovieLens website
genres = {'Action':0, 'Adventure':1, 'Animation':2, 'Comedy':3, 'Crime':4, 'Documentary':5, 'Drama':6, 'Family':7, 'Fantasy':8,
	  'Foreign':9, 'History':10, 'Horror':11, 'Music':12, 'Mystery':13, 'Romance':14, 'Science fiction':15, 'TV movie':16, 'Thriller':17, 
	  'War':18, 'Western':19, 'Children':20, 'Sci-Fi':21, 'IMAX':22, 'Musical':23, 'Film-Noir':24, 'Hirotaka Suzuoki':25 }

movie_genres = [[0 for i in range(len(genres))] for j in range(num_movies)] # stores movie genre information for each movie

firstline = True
count = 0

with open("../data/movies_map.pickle", "rb") as handle:
	u = pickle._Unpickler(handle)
	u.encoding = 'latin1'
	movies = u.load()

print("Movies loaded")

# ensure that your csv datafiles are inside a directory called "serendipity" outside the current directory
with open("../serendipity/movies.csv") as csvfile:
	filereader = csv.reader(csvfile, delimiter='\n', quotechar='|')
	
	for line in filereader:

		if count >= len(movies):
			break
		if firstline:
			firstline = False
			continue

		line = line[0]
		row = line.split(',')
		
		
		movieID = row[0]
		movieID = movieID[1:-1]
		
		## Extracting movie title information
		title = row[1]
		i = 2
		# the following lines of code are to handle the inconsistent nature of the text data
		# some of the movies have numbers in their titles. To distinguish this from the release date, we use the following crude hack
		if row[i][1].isdigit():
			while(True):
				# the following conditions have been hand-crafted after studying the dataset
				if row[i][1:-1][5:7] == '00' or row[i][1:-1][8:] == '00':
					break
				try:
					datetime.datetime.strptime(row[i][1:-1], '%Y-%m-%d')
					break
				except ValueError:
					i += 1
		else:
			while not row[i][1].isdigit():
				i += 1

		## Extracting release date information
		release_date = row[i]
		release_year = release_date[1:5]
		i += 1
		
		## Extracting director information
		director = []
		dirc = row[i]
		if dirc[-1] == '"':
			director.append(dirc[1:-1])
		else:
			director.append(dirc[1:])
		i += 1
		# To take care of the situations where there are more than one directors -- keep appending to the director list (as shown below)
		flag = False
		while dirc[-1] != '"':
			flag = True
			dirc = row[i]
			i += 1
			director.append(dirc)
		if flag:
			director.append(dirc[0:-1])
		

		## Extracting the movie cast information
		cast = []
		cast1 = row[i]
		cast.append(cast1[1:])
		
		while cast1[-1] != '"':
			i += 1
			cast1 = row[i]
			if cast1 == '':
				i += 1
				cast1 = row[i]
			cast.append(cast1)
		cast.append(cast1[0:-1])
		

		## Extracting genre information
		i += 3
		genre = []
		gr = row[i]
		genre.append(gr[1:])
		i += 1
		if i == len(row):
			continue
		gr = row[i]
		# to handle multiple genres for a movie
		while gr[-1] !='"':
			genre.append(gr)
			i += 1
			gr = row[i]
		genre.append(gr[0:-1])
  

		# handling the situation where a particular movie has not been chosen while creating our user-rating matrix
		if movieID not in movies.keys():
			continue
		else:
			count += 1
			movie_ind = movies[movieID]
		
			movie_date[movie_ind] = int(release_year)
			# for each movie we concatenate the directors and top 4 cast members and treat this new list as that movie's cast
			people = list(director)+cast[:4]
			c = collections.Counter(people)
			movie_cast[movie_ind] = c

			for g in genre:
				if g not in genres.keys():
					continue
				gen_ind = genres[g]
				movie_genres[movie_ind][gen_ind] = 1


# save the necessary files
with open('../data/movie_genres.pickle', 'wb') as handle:
	pickle.dump(movie_genres, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../data/movie_date.pickle', 'wb') as handle:
	pickle.dump(movie_date, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../data/movie_cast.pickle', 'wb') as handle:
	pickle.dump(movie_cast, handle, protocol=pickle.HIGHEST_PROTOCOL)



