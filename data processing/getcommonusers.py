### This code identifies the users taken into consideration for collaborative filtering from the answers.csv file of the dataset ###
### This is done in order to validate our results ###

import pickle
import csv
import numpy as np

firstline = True
count = 0
common = {}

# opening list of existing users taken into consideration
with open('../data/user_map.pickle', 'rb') as handle:
	users = pickle.load(handle)

with open("../serendipity/answers.csv") as csvfile:
	filereader = csv.reader(csvfile, delimiter='\n', quotechar='|')
	
	for line in filereader:
		count += 1
		
		if firstline:
			firstline = False
			continue

		line = line[0]
		row = line.split(',')

		userID = row[0]
		movieID = row[1]
		rating = row[2]

		if userID in users.keys():
			if userID not in common.keys():
				common[userID] = [tuple([movieID,float(rating)])]
			else:
				common[userID].append(tuple([movieID,float(rating)]))		


with open('../data/validation_users.pickle', 'wb') as handle:
	pickle.dump(common, handle, protocol=pickle.HIGHEST_PROTOCOL)

