"""
Content Based Filtering
"""

# Correlate the links to movies
links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies.dropna(subset=['id'], inplace=True)
movies['id'] = movies['id'].astype('int')

movies_l = movies[movies['id'].isin(links)]
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
print(tfidf_matrix)


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

crew['id'] = pd.to_numeric(crew['id'], errors='coerce')
crew.dropna(subset=['id'], inplace=True)
crew['id'] = crew['id'].astype('int')

movies = movies.merge(crew, on='id')
movies = movies.merge(keywords, on='id')

movies_k = movies[movies['id'].isin(links)]
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
def content_recommendations_improved(title,n):
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
    m = vote_counts.quantile(0)    # set the cutoff as 60%

    recommend = recommend_movies_improved[
        (recommend_movies_improved['vote_count'] >= m) & (recommend_movies_improved['vote_count'].notnull()) & (recommend_movies_improved['vote_average'].notnull())]
    recommend['vote_count'] = recommend['vote_count'].astype('int')
    recommend['vote_average'] = recommend['vote_average'].astype('float')
    recommend['WR'] = recommend.apply(weighted_rating, axis=1)
    recommend = recommend.sort_values('WR', ascending=False).head(n)
    return recommend


print(content_recommendations_improved('The Dark Knight Rises',n=10))

print(content_recommendations_improved('The Shawshank Redemption',n=10))
