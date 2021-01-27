#Gungor Yolac Project


import numpy as np
import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import SVD

#reading data from dataset.txt
columns = ['user_id', 'movie_id', 'rating']

datas = pd.read_csv('dataset.txt', sep=' ', names=columns)

a= {'userid': list(datas.user_id), 'itemid': list(datas.movie_id), 'rating': list(datas.rating)}

x = pd.DataFrame(a)

#ratings matrix
rats = datas.pivot(index='user_id',columns='movie_id',values='rating').fillna(0)

#print(datas)
#print(datas.iloc[1]['user_id'])

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(x[['userid', 'itemid', 'rating']], reader)

#divide the dataset
train_data, test_data = train_test_split(data, test_size=0.25)

#Training an SVD algorithm on the train_data.
algo = SVD()
algo.fit(train_data)

#Predicting ratings for all user-movie pairs that are NOT in the training set.
predictions = algo.test(test_data)


accuracy.rmse(predictions)

f = open("output.txt", "w")


#predicting each movie for each user
for i in range(1, 944):
    for mid in range(1, 1683):
        rat = int(algo.predict(uid=i, iid=mid).est)
        f.write(str(i) + ' ' + str(mid) + ' ' +  str(rat) + '\n')
