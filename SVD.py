# !pip install surprise

from surprise import Reader, Dataset, SVD
from surprise.model_selection import GridSearchCV, train_test_split
from surprise.accuracy import rmse
import pandas as pd

# Load the ratings data
reader = Reader()
ratings = pd.read_csv('ratings.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Define the parameter grid to search
param_grid = {'n_factors': [50, 100, 150],
              'n_epochs': [10, 20, 30],
              'lr_all': [0.002, 0.005, 0.01],
              'reg_all': [0.02, 0.04, 0.06]}

# Perform grid search with cross-validation
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1, return_train_measures=True)
gs.fit(data)

# Print the best RMSE score and the corresponding parameters
print('Best RMSE is:',gs.best_score['rmse'])
print('Best parameters are:',gs.best_params['rmse'])


# Train the model on the full dataset with the best parameters
svd = SVD(n_factors=gs.best_params['rmse']['n_factors'],
          n_epochs=gs.best_params['rmse']['n_epochs'],
          lr_all=gs.best_params['rmse']['lr_all'],
          reg_all=gs.best_params['rmse']['reg_all'])
trainset = data.build_full_trainset()
svd.fit(trainset)

# Make a prediction for user 1 and movie 302 with a rating of 3
prediction = svd.predict(1, 3671, 3)
print(prediction)


import pickle

# save the model
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd, f)

# read the model
with open('svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)
