import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
Item-Based Collaborative Filtering Recommendation
"""

# Build the rating matrix
ratings_matrix = movie_ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

# Calculate the similarity between movies
movie_similarity_matrix = 1 - pairwise_distances(ratings_matrix.T.values, metric='cosine')

print(movie_similarity_matrix.shape)
print(movie_similarity_matrix)

# Convert similarity matrix to DataFrame
movie_similarity = pd.DataFrame(movie_similarity_matrix, index=ratings_matrix.columns, columns=ratings_matrix.columns)


# Define function to get recommended movies
def get_recommendations_item(user_id, n):
    # Get unrated movies for the specified user
    user_ratings = ratings_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    # Calculate the recommendation score for each movie
    movie_scores = []
    for movie_id in unrated_movies:
        # Get the similarity between the movie and rated movies by the user
        similarity = movie_similarity[movie_id][user_ratings.index]
        # Calculate the recommendation score,
        # which is the sum of the product of similarity and rating divided by the sum of similarity
        score = np.sum(similarity * user_ratings) / np.sum(similarity)
        movie_scores.append((movie_id, score))
    # Sort movies by recommendation score and select the top n movies
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_movies = [movie[0] for movie in movie_scores[:n]]
    recommendations = movie_ratings.loc[movie_ratings['movieId'].isin(recommended_movies)]

    return recommendations[['movieId', 'title', 'genres']]

# Test
target_user_id = 1
recommendations = get_recommendations_item(target_user_id,n=10)
print(recommendations)