### This code is used to generate the profile of a user. A user profile is composed of a list of genre preferences ### 
### for each user in the dataset ####

import pickle
import numpy as np
import collections

with open('../data/movie_genres.pickle','rb') as handle:
	u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        genres = u.load()
print("Movie genres loaded")

with open('../data/movie_cast.pickle', 'wb') as handle:
	u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        cast = u.load()
print("Movie casts loaded")

with open('../data/user_rating_full.pickle','rb') as handle:
	user_rating = pickle.load(handle)
print("User ratings loaded")

num_users = len(user_rating)
user_genre_profile = [[] for i in range(num_users)]

for uid, ratings in enumerate(user_rating):
	
	user_genre = [0 for j in range(len(genres))]

	movie_count = 0
	for ind, movie_rating in enumerate(ratings):
		if movie_rating != 0:
			movie_count += 1
			# generating genre preference profile for each user
			g = genres[ind]
			#user_genre = np.array(user_genre) + np.array(g)
			user_genre[g] = 1

			# generating cast preference profile for each user
			c = cast[ind]
			user_cast = user_cast + c
	
	# populating the genre and cast preferences for each user
	user_genre_profile[uid] = np.divide(user_genre, movie_count)
	user_cast_profile[uid] = user_cast

with open('../data/user_genre_profiles.pickle','wb') as handle:
	pickle.dump(user_genre_profile, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../data/user_cast_profiles.pickle','wb') as handle:
	pickle.dump(user_cast_profile, handle, protocol=pickle.HIGHEST_PROTOCOL)

