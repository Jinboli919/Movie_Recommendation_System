import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# load the dataset
ratings_data = pd.read_csv('ratings.csv')
movies_data = pd.read_csv('movies_m.csv')
data = pd.merge(ratings_data, movies_data, on='movieId')

# create user and item input features
user_input = keras.Input(shape=(1,))
item_input = keras.Input(shape=(1,))

# create embedding layers
user_embedding = layers.Embedding(input_dim=data['userId'].max()+1, output_dim=50, input_length=1)(user_input)
item_embedding = layers.Embedding(input_dim=data['movieId'].max()+1, output_dim=50, input_length=1)(item_input)

# flatten the embedding layers
user_flatten = layers.Flatten()(user_embedding)
item_flatten = layers.Flatten()(item_embedding)

# concatenate the user and item features
concat = layers.Concatenate()([user_flatten, item_flatten])

# add dense layers for the neural network
dense1 = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(concat)
dense2 = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(dense1)
dense3 = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(dense2)
dense4 = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(dense3)

# add the output layer
output_layer = layers.Dense(1)(dense4)

# create the model
model = keras.Model(inputs=[user_input, item_input], outputs=output_layer)
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# split the data into training and testing datasets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# train the model
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
history = model.fit(x=[train_data['userId'], train_data['movieId']], y=train_data['rating'], epochs=10, batch_size=128, validation_data=([test_data['userId'], test_data['movieId']], test_data['rating']), callbacks=[reduce_lr])

# evaluate the model on the test data
results = model.evaluate([test_data['userId'], test_data['movieId']], test_data['rating'])
print(f'Test loss: {results[0]}, Test accuracy: {results[1]}')








































import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
ratings_data = pd.read_csv('ratings.csv')
movies_data = pd.read_csv('movies.csv')
movie_ratings = pd.merge(movies_data, ratings_data, on='movieId')

# Create User-Movie Rating Matrix
user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Split data into training and testing sets
train_data, test_data = train_test_split(user_movie_matrix, test_size=0.2, random_state=42)

# Calculate cosine similarity between users
user_similarity_matrix = cosine_similarity(train_data)

# Create the deep learning model
model = keras.Sequential([
    layers.Dense(128, input_shape=(train_data.shape[1],), activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(train_data.shape[1])
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(train_data, train_data, epochs=20, validation_data=(test_data, test_data), callbacks=[keras.callbacks.EarlyStopping(patience=5)])

# Make predictions for test data
user_prediction = model.predict(test_data)

# Calculate MAE and RMSE
print('MAE:', mean_absolute_error(test_data, user_prediction))
print('RMSE:', np.sqrt(mean_squared_error(test_data, user_prediction)))

# Defining a function to get K most similar users
def get_top_k_similar_users(target_user_id, k=5):
    target_user_similarity = user_similarity_matrix[target_user_id - 1]
    top_k_users = target_user_similarity.argsort()[::-1][1:k+1]
    top_k_similarity = target_user_similarity[top_k_users]
    return top_k_users, top_k_similarity

# Defining a function to get recommended movie list
def get_recommendations(target_user_id, k=5, n=10):
    top_k_users, _ = get_top_k_similar_users(target_user_id, k=k)
    neighbors_movies = user_movie_matrix.loc[top_k_users]
    target_user_movies = user_movie_matrix.loc[target_user_id][user_movie_matrix.loc[target_user_id] > 0].index
    neighbors_movies = neighbors_movies.loc[:, ~neighbors_movies.columns.isin(target_user_movies)]
    neighbors_mean_rating = neighbors_movies.mean(axis=0)
    neighbors_mean_rating = neighbors_mean_rating.sort_values(ascending=False)
    recommendations = movies_data.loc[movies_data['title'].isin(neighbors_mean_rating.head(n).index)]
    return recommendations[['title', 'genres']]

# Test
target_user_id = 1
recommendations = get_recommendations(target_user_id, k=5, n=10)
print(recommendations)




























def get_recommendations(target_user_id, n=10):
    # create a dataframe of all the movies in the dataset
    all_movies = pd.DataFrame(data['movieId'].unique(), columns=['movieId'])

    # create a dataframe of all the movies rated by the target user
    rated_movies = data[data['userId'] == target_user_id]['movieId']

    # create a dataframe of all the movies the target user has not rated
    unrated_movies = all_movies[~all_movies['movieId'].isin(rated_movies)]

    # create a dataframe of the target user and the unrated movies
    target_user = pd.DataFrame({'userId': [target_user_id] * len(unrated_movies), 'movieId': unrated_movies['movieId']})

    # use the model to predict the ratings for the target user and unrated movies
    target_user['predicted_rating'] = model.predict([target_user['userId'], target_user['movieId']])

    # sort the predicted ratings in descending order and select the top n movies
    top_n = target_user.sort_values(by='predicted_rating', ascending=False).head(n)

    # merge the top_n dataframe with the movies_data dataframe to get the movie titles
    recommendations = pd.merge(top_n, movies_data, on='movieId')[['title', 'genres']]

    # return the list of recommended movies
    return recommendations


target_user_id = 3
recommendations = get_recommendations(target_user_id, n=10)
print(recommendations)