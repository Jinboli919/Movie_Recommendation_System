import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
Import Data
"""

ratings = pd.read_csv(r'D:\Capstone\dataset\ratings.csv')
links = pd.read_csv(r'D:\Capstone\dataset\links.csv')
keywords = pd.read_csv(r'D:\Capstone\dataset\keywords.csv')
credits = pd.read_csv(r'D:\Capstone\dataset\credits.csv')
movies_m = pd.read_csv(r'D:\Capstone\dataset\movies_m.csv')

movie_ratings = pd.merge(movies_m, ratings, on='movieId')
print(movie_ratings)

# Building Movie-User Rating Matrix
movie_user_matrix = movie_ratings.pivot_table(index='movieId', columns='userId', values='rating')
print(movie_user_matrix)
# pivot table of movie-user ratings
movie_ratings_pivot = movie_ratings.pivot_table(index='title', columns='userId', values='rating')

# split data into training and testing sets
train_data, test_data = train_test_split(movie_ratings_pivot, test_size=0.2, random_state=42)

# calculate cosine similarity between movies
item_similarity = cosine_similarity(train_data.fillna(0).T)

print(item_similarity)

# function to make predictions using item-based collaborative filtering
def predict(ratings, similarity):
    pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# make predictions for test data
item_prediction = predict(train_data.fillna(0), item_similarity)

# reindex item_prediction to align with test_data
item_prediction = pd.DataFrame(item_prediction, index=train_data.columns, columns=train_data.index).T
item_prediction = item_prediction.reindex(test_data.index)

# calculate MAE and RMSE
print('MAE:', mean_absolute_error(test_data.fillna(0), item_prediction.fillna(0)))
print('RMSE:', np.sqrt(mean_squared_error(test_data.fillna(0), item_prediction.fillna(0))))
item_similarity_matrix = cosine_similarity(movie_user_matrix.fillna(0).T)

# Defining a function to get K most similar movies
def get_top_k_similar_movies(target_movie_id, k=5):
    target_movie_similarity = item_similarity_matrix[target_movie_id - 1]
    top_k_movies = target_movie_similarity.argsort()[::-1][1:k+1]
    top_k_similarity = target_movie_similarity[top_k_movies]
    return top_k_movies, top_k_similarity

# Defining a function to get recommended movie list
def get_recommendations(target_user_id, k=5, n=10):
    top_k_users, _ = get_top_k_similar_movies(target_user_id, k=k)
    neighbors_movies = movie_user_matrix.loc[top_k_users]
    target_user_movies = movie_user_matrix.loc[target_user_id].dropna().index
    neighbors_movies = neighbors_movies.loc[:, ~neighbors_movies.columns.isin(target_user_movies)]
    neighbors_mean_rating = neighbors_movies.mean(axis=0)
    neighbors_mean_rating = neighbors_mean_rating.sort_values(ascending=False)
    recommendations = movies_m.loc[movies_m['movieId'].isin(neighbors_mean_rating.head(n).index)]
    return recommendations[['title', 'genres']]

# Test
target_user_id = 1
recommendations = get_recommendations(target_user_id, k=5, n=10)
print(recommendations)
