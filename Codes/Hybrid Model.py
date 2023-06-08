"""
Hybrid Model
"""


with open('svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)

# link the movieId to id in the movies.csv
integrate_id = pd.read_csv('links_small.csv')[['movieId', 'tmdbId']]

# define a function to convert 'tmdbId' values to integers
def convert_id(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return np.nan

integrate_id['tmdbId'] = integrate_id['tmdbId'].apply(convert_id)
integrate_id.columns = ['movieId', 'id']

# merge the 'integrate_id' and 'movies_k' dataframes on the 'id' column and set the index to 'title'
integrate_id = integrate_id.merge(movies_k[['title', 'id']], on='id').set_index('title')

# create a dictionary that maps the 'id' column to its corresponding index in the 'cosine_sim' matrix
indices_integrate = integrate_id.set_index('id')

def hybrid(userId, title, n):
    idx = indices[title]
    # get the 'tmdbId' and 'movieId' of the input movie title
    tmdbId = integrate_id.loc[title]['id']
    movie_id = integrate_id.loc[title]['movieId']

    # compute the cosine similarities between the input movie and all other movies
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    # sort the similarities in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # select the top 30 most similar movies
    sim_scores = sim_scores[1:31]
    # get the indices of the most similar movies
    movie_indices = [i[0] for i in sim_scores]

    # select the relevant columns from the 'movies_k' dataframe for the most similar movies
    movies = movies_k.iloc[movie_indices][['id', 'title', 'vote_count', 'vote_average', 'genres', 'director', 'cast', 'year', 'keywords']]
    movies['movieId'] = movies['id']
    movies = movies.drop(columns=['id'])
    # predict the rating of the most similar movies for the input user using the SVD model
    movies['Predicted rating'] = movies['movieId'].apply(lambda x: svd.predict(userId, indices_integrate.loc[x]['movieId']).est)
    # sort the movies in descending order of predicted rating
    movies = movies[['movieId', 'title', 'vote_count', 'vote_average', 'genres', 'director', 'cast', 'year', 'keywords', 'Predicted rating']].sort_values('Predicted rating', ascending=False)

    return movies.head(n)

# Test
hybrid(98, 'The Dark Knight Rises',20)