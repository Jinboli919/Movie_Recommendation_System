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

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

"""
Import Data
"""

movies = pd.read_csv(r'D:\Capstone\dataset\movies.csv')
ratings = pd.read_csv(r'D:\Capstone\dataset\ratings.csv')
links = pd.read_csv(r'D:\Capstone\dataset\links.csv')
keywords = pd.read_csv(r'D:\Capstone\dataset\keywords.csv')
credits = pd.read_csv(r'D:\Capstone\dataset\credits.csv')
movies_m = pd.read_csv(r'D:\Capstone\dataset\movies_m.csv')
print(movies.head())
print(ratings.head())
print(links.head())
print(keywords.head())


"""
Data Pre-processing and EDA
"""

"""
Clean the genres data and extract their names
"""

def get_genre_names(genre_list: List[dict]) -> List[str]:
    """
    Extracts the names of movie genres.
    """
    return [i['name'] for i in genre_list]

def clean_genre_column(df, column_name: str):
    """
    Cleans a column in a DataFrame by extracting the genre names for each movie.
    """
    df[column_name] = ( df[column_name].fillna('[]')   # Fill missing values with an empty list
        .apply(ast.literal_eval)       # Convert strings to Python lists
        .apply(get_genre_names) )      # Extract the genre names

# Call the function to clean the 'genres' column
clean_genre_column(movies, 'genres')
print(movies['genres'].head(10))


"""
Extract the release year
"""

def extract_year(date_str):
    """
    Extract the release year from the release date
    """
    if date_str != np.nan:
        return str(pd.to_datetime(date_str, errors='coerce').year)
    else:
        return np.nan

movies['year'] = movies['release_date'].apply(extract_year)






"""
Draw a histogram plot to show the most prolific movie genres
"""

# Define function to get counts of each category in a column
def get_counts(df, column_name: str, categories: list):
    """
    Returns the count of each category in a column of a DataFrame.
    """
    counts = {}
    for cat in categories:
        counts[cat] = df[column_name].apply(lambda x: cat in x).sum()
    return counts

# Get the base counts for each category and sort them by counts
genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama',
          'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music', 'Mystery', 'Romance',
          'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

base_counts = get_counts(movies, 'genres', genres)
base_counts = pd.DataFrame(index=base_counts.keys(),
                           data=base_counts.values(),
                           columns=['Counts'])
base_counts.sort_values(by='Counts', inplace=True)

# Plot the chart which shows top genres and separate by color where genre < 3000
colors=['#a5a5a5' if i<3000 else '#2ecc71' for i in  base_counts.Counts]

plt.figure(figsize=(12,8))
plt.bar(x=base_counts.index, height=base_counts.Counts, color=colors)
plt.title('Most prolific movie Genres')
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.25)
plt.xlabel('Genres')
plt.ylabel('Numbers of movies')

# Add the color legend of counts
legend_elements = [Patch(facecolor='#a5a5a5', label='Counts < 3000'),
                   Patch(facecolor='#2ecc71', label='Counts >= 3000')]
plt.legend(handles=legend_elements, loc='upper left')
plt.show()


"""
Draw a histogram plot to show the movies released per year
"""
# Group movies by year and count the number of movies released per year
movies_per_year = movies.groupby('year')['title'].count().reset_index()

# Create a bar plot with a gradient color scheme
ax = sns.barplot(data=movies_per_year, x='year', y='title', palette='viridis')

plt.title('Movies Released per Year')
ax.set_xlabel('Year')
plt.ylabel('Number of Movies')

# Set the x-axis tick locations and labels
xtick_locs = np.arange(0, len(movies_per_year), 20)
xtick_labels = [str(x) for x in range(1890, 2022, 20)]
plt.xticks(xtick_locs, xtick_labels)

# Add the colorbar to the right side of the plot
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=1890, vmax=2020))
sm._A = []
cbar = plt.colorbar(sm)

plt.show()



"""
Draw a pie chart to show the top 15 popular genres by vote_count
"""
# Create a new DataFrame with the vote count and rating information
genres_df = movies[['genres', 'vote_count', 'vote_average']].explode('genres')
genres_df = genres_df.groupby('genres').agg({'vote_count': 'sum', 'vote_average': 'mean'}).reset_index()

# Get the top 15 genres by vote count
top_genres = genres_df.sort_values('vote_count', ascending=False).head(15)

# Create a pie chart of the top genres by vote count
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(top_genres['vote_count'], colors=[f'C{i}' for i in range(len(top_genres))], autopct='%1.1f%%')

# Create a legend for the genres' colors
handles = []
for i, genre in enumerate(top_genres['genres']):
    handles.append(mpatches.Patch(color=f"C{i}", label=genre))
plt.legend(handles=handles, bbox_to_anchor=(1.1, 0.8), loc='upper left', fontsize=12)

# Show the genres' names with the percentages in the legend
legend_labels = []
for i in range(len(texts)):
    label = f"{top_genres.iloc[i]['genres']}: {autotexts[i].get_text()}"
    legend_labels.append(label)
ax.legend(wedges, legend_labels, bbox_to_anchor=(1.25, 1), title='Genres', loc='upper right', fontsize=10)

plt.title('Top 15 Popular Genres by Vote_count')
plt.show()


"""
Draw a bar chart to show the top 15 genres with the highest average ratings by vote_average.
"""
# Get the top 15 genres by vote average
top_genres_a = genres_df.sort_values('vote_average', ascending=False).head(15)
top_genres_a = top_genres_a.sort_values('vote_average', ascending=True)

# Create a horizontal bar chart of the top genres by vote average
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_genres_a['genres'], top_genres_a['vote_average'], color=[f'C{i}' for i in range(len(top_genres))])

plt.title('Top 15 Genres with the Highest Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Genres')
plt.show()


"""
Draw a bar chart to show the top 15 popular genres by popularity
"""
# Clean the popularity column
movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce').fillna(0)

# Create a new DataFrame with the vote count and popularity information
genres_df = movies[['genres', 'popularity']].explode('genres')
genres_df = genres_df.groupby('genres').agg({'popularity': 'mean'}).reset_index()

# Get the top 15 genres by popularity and sort them in descending order
top_genres = genres_df.sort_values('popularity', ascending=False).head(15)
top_genres = top_genres.sort_values('popularity', ascending=True)

# Create a bar chart of the top genres by popularity
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_genres['genres'], top_genres['popularity'], color=[f'C{i}' for i in range(len(top_genres))])

plt.title('Top 15 Popular Genres by Popularity')
plt.xlabel('Popularity')
plt.show()




















"""
Top 50 movies recommendation in different popular genres
"""
# Build the overall Top 50 movies recommendation chart in different popular genres.

# Create a Weighted rating formula to get the result chart.
# Weighted Rating = [(v/v+m)*R] + [(m/v+m)*C]
# v is the number of votes for a movie. This is an indicator of how popular or well-known the movie is.
# m is the minimum number of votes required for a movie to be listed in the chart.
# R is the average rating of the movie.
# C is the mean vote across all movies in the dataset.

# Clean and convert the vote_count and vote_average column
vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('float')

# Calculate C
C = vote_averages.mean()

# Use 90% as the cutoff to calculate m
m = vote_counts.quantile(0.90)

# To get the qualified movies
Top_movies = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull())
            & (movies['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]

# Convert vote_count to int and vote_average to float
Top_movies['vote_count'] = Top_movies['vote_count'].astype('int')
Top_movies['vote_average'] = Top_movies['vote_average'].astype('float')

# Create WR formula
def weighted_rating(x):
    """
    Calculates the weighted rating for a given movie based on its vote count,
    vote average, and the total mean vote and minimum vote count required.
    """
    # Extract the vote count and vote average for the movie
    v = x['vote_count']
    R = x['vote_average']

    # Calculate the weighted rating using the formula
    WR = (v / (v + m) * R) + (m / (m + v) * C)

    return WR

Top_movies['WR'] = Top_movies.apply(weighted_rating,axis = 1)
Top_movies = Top_movies.sort_values('WR', ascending= False).head(50)
print(Top_movies)


"""
Get the top 20 movies in some popular genres
"""

# We can use the WR formula to get the top 20 movies in different popular genres, like comedy, action and so on
# A movie may belong to different movie genres, so we need to split the genres column

def split_genres(movies):
    """
    split the genres into multiple rows
    and return a new one with one row per genre.
    """
    # Split the genres column into multiple rows using lambda function and stack() method
    s = movies.apply(lambda x: pd.Series(x['genres'], dtype='str'), axis=1).stack()

    # Drop the original index level and rename the series
    s = s.reset_index(level=1, drop=True).rename('genre')

    # Drop the genres column from the original dataframe and join with the new series
    movies_genre = movies.drop('genres', axis=1).join(s)

    return movies_genre

movies_genre = split_genres(movies)
# print(movies_genre)

# Create a function to get the top 20 movies in different genres with the new cutoff with 80%
def build_top(genre, percentile=0.8, genre_name = None):
    # Select all movies of the given genre from the preprocessed movies_genre
    genre_movies = movies_genre[movies_genre['genre'] == genre]

    # Calculate the mean vote average and the vote count threshold
    vote_averages = genre_movies[genre_movies['vote_average'].notnull()]['vote_average'].astype('float')
    C = vote_averages.mean()
    vote_counts = genre_movies[genre_movies['vote_count'].notnull()]['vote_count'].astype('int')
    m = vote_counts.quantile(percentile)

    # To get the qualified movies
    Top_20 = genre_movies[(genre_movies['vote_count'] >= m) & (genre_movies['vote_count'].notnull()) & (
        genre_movies['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]

    # Convert vote_count to int and vote_average to float
    Top_20['vote_count'] = Top_20['vote_count'].astype('int')
    Top_20['vote_average'] = Top_20['vote_average'].astype('float')

    # Calculate the weighted rating using the formula
    Top_20['WR'] = Top_20.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (m / (m + x['vote_count']) * C),
        axis=1)

    Top_20 = Top_20.sort_values('WR', ascending=False).head(20)

    if genre_name:
        Top_20['genre'] = genre_name

    return Top_20


"""
Try to return the top 20 movies in some popular genres
"""

Action_20 = build_top('Action', genre_name='Action')
print(Action_20)

Drama_20 = build_top('Drama', genre_name='Drama')
print(Drama_20)

Adventure_20 = build_top('Adventure', genre_name='Adventure')
print(Adventure_20)



"""
Content Based Filtering 
"""

# Correlate the links to movies
links_small = pd.read_csv(r'D:\Capstone\dataset\links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies.dropna(subset=['id'], inplace=True)
movies['id'] = movies['id'].astype('int')

movies_l = movies[movies['id'].isin(links_small)]
print(movies_l)

# 1. Using the descriptions and taglines of movies to recommend
movies_l['tagline'] = movies_l['tagline'].fillna('')
movies_l['description'] = movies_l['overview'] + movies_l['tagline']
movies_l['description'] = movies_l['description'].fillna('')

# Create a TfidfVectorizer object
tf = TfidfVectorizer(
    analyzer='word',     # Analyze at the word level
    ngram_range=(1, 2),  # Include unigrams and bigrams
    min_df=0,            # Include words that occur in at least 1 document
    stop_words='english' # Exclude English stop words
)

# Use the TfidfVectorizer to transform the 'description' column of the movies_l dataset
# into a matrix of tf-idf features
tfidf_matrix = tf.fit_transform(movies_l['description'])

print(tfidf_matrix.shape)


# Use dot product to get the Cosine Similarity.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Write a function to return the 15 most similar movies based on Cosine Similarity score
movies_l = movies_l.reset_index()
titles = movies_l['title']
indices = pd.Series(movies_l.index, index=movies_l['title'])


def content_recommendations(title):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select the top 15 most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:11]

    # Get the indices of the selected movies
    movie_indices = [i[0] for i in sim_scores]

    recommend_movies = movies_l.iloc[movie_indices][['title', 'vote_average', 'genres', 'year', 'description']]

    return recommend_movies

print(content_recommendations('The Dark Knight Rises'))

print(content_recommendations('The Shawshank Redemption'))



# 2. Using the crew, cast and keywords to recommend

# Correlate the credits and keywords to movies
# Clean the id in keywords and credits
keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')
keywords.dropna(subset=['id'], inplace=True)
keywords['id'] = keywords['id'].astype('int')

credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
credits.dropna(subset=['id'], inplace=True)
credits['id'] = credits['id'].astype('int')

movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')

movies_k = movies[movies['id'].isin(links_small)]
print(movies_k.shape)



movies_k['cast'] = movies_k['cast'].apply(literal_eval)
movies_k['crew'] = movies_k['crew'].apply(literal_eval)
movies_k['keywords'] = movies_k['keywords'].apply(literal_eval)
movies_k['cast_size'] = movies_k['cast'].apply(lambda x: len(x))
movies_k['crew_size'] = movies_k['crew'].apply(lambda x: len(x))

# Get the director name
def get_director(crew_list):
    for crew_member in crew_list:

        if crew_member['job'] == 'Director':

            return crew_member['name']

    return np.nan

movies_k['director'] = movies_k['crew'].apply(get_director)


def get_actor_names(cast_list):
    if isinstance(cast_list, list):
        # Extract actor names from the list
        return [i['name'] for i in cast_list]
    else:
        # Return an empty list for non-list entries
        return []

movies_k['cast'] = movies_k['cast'].apply(get_actor_names)

def limit_actors(cast_list, max_actors=3):
    if len(cast_list) >= max_actors:
        # Return the top 3 actors
        return cast_list[:max_actors]
    else:
        # Return the entire list for less than 3 actors
        return cast_list

movies_k['cast'] = movies_k['cast'].apply(limit_actors)

def get_keywords(keywords_list):
    if isinstance(keywords_list, list):
        # Extract keyword names from the list
        return [i['name'] for i in keywords_list]
    else:
        # Return an empty list for non-list entries
        return []

movies_k['keywords'] = movies_k['keywords'].apply(get_keywords)


def clean_cast_names(cast_list):
    """
    Strip name spaces and convert to lowercase
    """
    cleaned_list = []
    for name in cast_list:
        cleaned_name = str.lower(name.replace(" ", ""))
        cleaned_list.append(cleaned_name)
    return cleaned_list

movies_k['cast'] = movies_k['cast'].apply(clean_cast_names)


def clean_director_name(name):
    """
    Strip  directors' name spaces and convert to lowercase
    """
    return str.lower(name.replace(" ", ""))

def add_director_weight(director_name):
    """
    Count director name 3 times to give it more weight relative to the cast
    """
    return [director_name, director_name, director_name,]

movies_k['director'] = movies_k['director'].astype('str').apply(clean_director_name)
movies_k['director'] = movies_k['director'].apply(add_director_weight)


def extract_keywords(dataset):
    """
    Extract keywords and reorganize them into a new dataset
    """
    keywords = dataset.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
    keywords.name = 'keywords'
    return keywords

e_keywords = extract_keywords(movies_k)

# Remove the keywords which only occur once
e_keywords = e_keywords.value_counts()
e_keywords = e_keywords[e_keywords > 1]

# Using SnowballStemmer to reduce a word to its base or root form
def stem_keywords(keyword_list):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(keyword) for keyword in keyword_list]

# Create a function to filter keywords
def filter_keywords(keywords):
    words = []
    for i in keywords:
        if i in e_keywords:
            words.append(i)
    return words

# Strip keywords the spaces and convert to lowercase
def clean_keywords(keyword_list):
    return [str.lower(keyword.replace(" ", "")) for keyword in keyword_list]

movies_k['keywords'] = movies_k['keywords'].apply(filter_keywords)
movies_k['keywords'] = movies_k['keywords'].apply(stem_keywords)
movies_k['keywords'] = movies_k['keywords'].apply(clean_keywords)

movies_k['g_features'] = movies_k['genres'] + movies_k['director'] + movies_k['cast'] + movies_k['keywords']
movies_k['g_features'] = movies_k['g_features'].apply(lambda x: ' '.join(x))

# Create a new Count Vectorizer
cf = CountVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    min_df=0,
    stop_words='english'
)

count_matrix = cf.fit_transform(movies_k['g_features'])

# Get the Cosine Similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# Create a funtion to recommend based on general features
movies_k = movies_k.reset_index()
titles = movies_k['title']
indices = pd.Series(movies_k.index, index=movies_k['title'])


def content_recommendations_g(title):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select the top 15 most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:11]

    # Get the indices of the selected movies
    movie_indices = [i[0] for i in sim_scores]

    recommend_movies = movies_k.iloc[movie_indices][['title', 'vote_average', 'genres', 'director', 'cast', 'year', 'keywords']]
    # Only return once of the director's name
    recommend_movies['director'] = recommend_movies['director'].apply(lambda x: list(set(x))[0])

    return recommend_movies


print(content_recommendations_g('The Dark Knight Rises'))

print(content_recommendations_g('The Shawshank Redemption'))




# 3. Improve the content-based recommendation by adding indexs of pipolarity and ratings
def content_recommendations_improved(title):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select the top 15 most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:16]

    # Get the indices of the selected movies
    movie_indices = [i[0] for i in sim_scores]

    recommend_movies_improved = movies_k.iloc[movie_indices][
        ['title', 'vote_count', 'vote_average', 'genres', 'director', 'cast', 'year', 'keywords']]
    # Only return once of the director's name
    recommend_movies_improved['director'] = recommend_movies_improved['director'].apply(lambda x: list(set(x))[0])
    vote_counts = recommend_movies_improved[recommend_movies_improved['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = recommend_movies_improved[recommend_movies_improved['vote_average'].notnull()]['vote_average'].astype('float')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)    # set the cutoff as 60%

    recommend = recommend_movies_improved[
        (recommend_movies_improved['vote_count'] >= m) & (recommend_movies_improved['vote_count'].notnull()) & (recommend_movies_improved['vote_average'].notnull())]
    recommend['vote_count'] = recommend['vote_count'].astype('int')
    recommend['vote_average'] = recommend['vote_average'].astype('float')
    recommend['WR'] = recommend.apply(weighted_rating, axis=1)
    recommend = recommend.sort_values('WR', ascending=False).head(6)
    return recommend


print(content_recommendations_improved('The Dark Knight Rises'))

print(content_recommendations_improved('The Shawshank Redemption'))














"""
Collaborative Filtering
"""


"""
User-Based Collaborative Filtering Recommendation
"""

movie_ratings = pd.merge(movies_m, ratings,on='movieId')
print(movie_ratings)

# Building User-Movie Rating Matrix
user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='movieId', values='rating')

# pivot table of user-movie ratings
movie_ratings_pivot = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# split data into training and testing sets
train_data, test_data = train_test_split(movie_ratings_pivot, test_size=0.2, random_state=42)

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
    target_user_movies = user_movie_matrix.loc[target_user_id][user_movie_matrix.loc[target_user_id] > 0].index
    neighbors_movies = neighbors_movies.loc[:, ~neighbors_movies.columns.isin(target_user_movies)]
    neighbors_mean_rating = neighbors_movies.mean(axis=0)
    neighbors_mean_rating = neighbors_mean_rating.sort_values(ascending=False)
    recommendations = movies_m.loc[movies_m['movieId'].isin(neighbors_mean_rating.head(n).index)]

    return recommendations[['title', 'genres']]




# Test
target_user_id = 1
recommendations = get_recommendations(target_user_id, k=5, n=10)
print(recommendations)




"""
Item-Based Collaborative Filtering Recommendation
"""


movie_ratings = pd.merge(movies_m, ratings, on='movieId')
print(movie_ratings)

# Building Movie-User Rating Matrix
movie_user_matrix = movie_ratings.pivot_table(index='movieId', columns='userId', values='rating')

# pivot table of movie-user ratings
movie_ratings_pivot = movie_ratings.pivot_table(index='title', columns='userId', values='rating')

# split data into training and testing sets
train_data, test_data = train_test_split(movie_ratings_pivot, test_size=0.2, random_state=42)

# calculate cosine similarity between movies
item_similarity = cosine_similarity(train_data.fillna(0).T)

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

