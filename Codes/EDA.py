# !pip install surprise
import pandas as pd
from typing import List
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


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
Draw a heatmap to show the movies releases by month and year in this century
"""

movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def get_month(x):
    try:
        return month_order[int(str(x).split('-')[1]) - 1]
    except:
        return np.nan

def get_day(x):
    try:
        year, month, day = (int(i) for i in x.split('-'))
        answer = datetime.date(year, month, day).weekday()
        return day_order[answer]
    except:
        return np.nan


movies['day'] = movies['release_date'].apply(get_day)
movies['month'] = movies['release_date'].apply(get_month)

movies[movies['year'] != 'NaT'][['title', 'year']].sort_values('year').head(10)

months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
movies1 = movies.copy()
movies1['year'] = movies1[movies1['year'] != 'NaT']['year'].astype(int)
movies1 = movies1[movies1['year'] >=2000]
heatmap = pd.pivot_table(data=movies1, index='month', columns='year', aggfunc='count', values='title')
heatmap = heatmap.fillna(0)

sns.set(font_scale=1)
f, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(heatmap, annot=True, linewidths=.5, ax=ax, cmap='YlOrRd', fmt='n', yticklabels=month_order, xticklabels=heatmap.columns.astype(int))
ax.set_title('Heatmap of Movie Releases by Month and Year', fontsize=20, fontweight='bold')
plt.xticks(rotation=0)
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
Draw a wordcloud plot to show the most common words in movie titles
"""

movies['title'] = movies['title'].astype('str')
movies['overview'] = movies['overview'].astype('str')

title_corpus = ' '.join(movies['title'])
overview_corpus = ' '.join(movies['overview'])


title_wordcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='white',
    colormap='Set2',
    height=600,
    width=800
).generate(title_corpus)


plt.figure(figsize=(12,6))
plt.imshow(title_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Movie Titles Wordcloud", fontsize=20, fontweight='bold', color='darkblue')
plt.show()


"""
Draw a wordcloud plot to show the most common words in movie overviews
"""

overview_wordcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='white',
    colormap='Set2',
    height=600,
    width=800
).generate(overview_corpus)

plt.figure(figsize=(12,6))
plt.imshow(overview_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Movie Overview Wordcloud", fontsize=20, fontweight='bold', color='darkblue')
plt.show()


"""
Draw a scatter plot to show the distribution of movie mean rating and the number of ratings
"""

data = pd.merge(ratings, movies_m, on='movieId')

agg_ratings = data.groupby('title').agg(mean_rating = ('rating', 'mean'),
                    number_of_ratings = ('rating', 'count')).reset_index()

plot = sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings)
plot.set_axis_labels("Mean Rating", "Number of Ratings", fontsize=10)
plot.fig.suptitle("Distribution of Mean Rating and Number of Ratings", fontsize=12)
plot.fig.subplots_adjust(top=0.95)
