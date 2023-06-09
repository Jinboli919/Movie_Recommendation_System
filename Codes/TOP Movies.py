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