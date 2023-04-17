import pandas as pd
from typing import List
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

"""
Import Data
"""


ratings = pd.read_csv(r'D:\Capstone\dataset\ratings.csv')
links = pd.read_csv(r'D:\Capstone\dataset\links.csv')
keywords = pd.read_csv(r'D:\Capstone\dataset\keywords.csv')
credits = pd.read_csv(r'D:\Capstone\dataset\credits.csv')
movies_m = pd.read_csv(r'D:\Capstone\dataset\movies_m.csv')


"""
User-Based Collaborative Filtering Recommendation
"""

movie_ratings = pd.merge(movies_m, ratings,on='movieId')
print(movie_ratings)

# Building User-Movie Rating Matrix
user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='movieId', values='rating')
print(user_movie_matrix)
# pivot table of user-movie ratings
movie_ratings_pivot = movie_ratings.pivot_table(index='userId', columns='title', values='rating')
print(movie_ratings_pivot)

# split data into training and testing sets
train_data, test_data = train_test_split(movie_ratings_pivot, test_size=0.2, random_state=42)


print(train_data)
# calculate cosine similarity between users
user_similarity = cosine_similarity(train_data.fillna(0))

print(user_similarity)

# function to make predictions using user-based collaborative filtering
def predict(ratings, similarity):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred

# make predictions for test data
user_prediction = predict(train_data.fillna(0), user_similarity)

# reindex user_prediction to align with test_data
user_prediction = pd.DataFrame(user_prediction, index=train_data.index, columns=train_data.columns)
user_prediction = user_prediction.reindex(test_data.index)

# calculate MAE and RMSE
print('MAE:', mean_absolute_error(test_data.fillna(0), user_prediction.fillna(0)))
print('RMSE:', np.sqrt(mean_squared_error(test_data.fillna(0), user_prediction.fillna(0))))
user_similarity_matrix = cosine_similarity(user_movie_matrix.fillna(0))

# Defining a function to get K most similar users
def get_top_k_similar_users(target_user_id, k=5):
    target_user_similarity = user_similarity_matrix[target_user_id - 1]
    top_k_users = target_user_similarity.argsort()[::-1][1:k+1]
    top_k_similarity = target_user_similarity[top_k_users]
    return top_k_users, top_k_similarity

def get_recommendations(target_user_id, k=5, n=10):
    top_k_users, _ = get_top_k_similar_users(target_user_id, k=k)
    neighbors_movies = user_movie_matrix.loc[top_k_users]
    target_user_movies = user_movie_matrix.loc[target_user_id][user_movie_matrix.loc[target_user_id] > 0.3].index
    neighbors_movies = neighbors_movies.loc[:, ~neighbors_movies.columns.isin(target_user_movies)]
    neighbors_mean_rating = neighbors_movies.mean(axis=0)
    neighbors_mean_rating = neighbors_mean_rating.sort_values(ascending=False)
    recommendations = movies_m.loc[movies_m['movieId'].isin(neighbors_mean_rating.head(n).index)]
    return recommendations[['title', 'genres']]




# Test
target_user_id = 1
recommendations = get_recommendations(target_user_id, k=5, n=10)
print(recommendations)






