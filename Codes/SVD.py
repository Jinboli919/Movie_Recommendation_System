# !pip install surprise

from surprise import Reader, Dataset, SVD
from surprise.model_selection import GridSearchCV, train_test_split
from surprise.accuracy import rmse
import pandas as pd

"""
SVD (Singular Value Decomposition) Approach
"""

from surprise import Reader, Dataset, SVD
from surprise.model_selection import GridSearchCV, train_test_split
from surprise.accuracy import rmse, mae
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


reader = Reader()
ratings = pd.read_csv('ratings.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

# Define the parameter grid to search
param_grid = {'n_factors': [10, 30, 50, 70, 90, 100, 150],
              'n_epochs': [30, 40, 50],
              'lr_all': [0.001, 0.005, 0.01, 0.1],
              'reg_all': [0.01, 0.02, 0.04, 0.06]}

# Perform grid search with cross-validation
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1, return_train_measures=True)
gs.fit(data)

# Print the best RMSE score and the corresponding parameters
print('Best RMSE is:',gs.best_score['rmse'])
print('Best MAE is:',gs.best_score['mae'])
print('Best parameters are:',gs.best_params['rmse'])

# Extract the results from the grid search
results = gs.cv_results
factors = np.array(param_grid['n_factors'])
epochs = np.array(param_grid['n_epochs'])
lrs = np.array(param_grid['lr_all'])
regs = np.array(param_grid['reg_all'])

# Compute the precision scores
precision_scores = []
for params, mean_test_score, _ in zip(results['params'], results['mean_test_rmse'], results['std_test_rmse']):
    svd = SVD(**params)
    svd.fit(trainset)
    predictions = svd.test(testset)
    y_true = [int(pred.r_ui >= 3) for pred in predictions]
    y_pred = [int(pred.est >= 3) for pred in predictions]
    score = f1_score(y_true, y_pred)
    precision_scores.append(score)

# Reshape and plot the precision scores
precision_scores = np.array(precision_scores)
precision_scores = np.reshape(precision_scores, (len(factors), len(epochs), len(lrs), len(regs)))
precision_scores = np.mean(precision_scores, axis=(2, 3))
for i, epoch in enumerate(epochs):
    plt.plot(factors, precision_scores[:,i], label=f'n_epochs={epoch}')
plt.title('SVD Performance: Precision Against Number of Factors')
plt.xlabel('n_factors')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Plot the RMSE scores
rmse_scores = results['mean_test_rmse']
rmse_scores = np.reshape(rmse_scores, (len(factors), len(epochs), len(lrs), len(regs)))
rmse_scores = np.mean(rmse_scores, axis=(2, 3))
for i, epoch in enumerate(epochs):
    plt.plot(factors, rmse_scores[:,i], label=f'n_epochs={epoch}')
plt.title('SVD Performance: RMSE Against Number of Factors')
plt.xlabel('n_factors')
plt.ylabel('RMSE')
plt.legend()
plt.show()

# Plot the MAE scores
mae_scores = results['mean_test_mae']
mae_scores = np.reshape(mae_scores, (len(factors), len(epochs), len(lrs), len(regs)))
mae_scores = np.mean(mae_scores, axis=(2, 3))
for i, epoch in enumerate(epochs):
    plt.plot(factors, mae_scores[:,i], label=f'n_epochs={epoch}')
plt.title('SVD Performance: MAE Against Number of Factors')
plt.xlabel('n_factors')
plt.ylabel('MAE')
plt.legend()
plt.show()


# Train the model on the full dataset with the best parameters
svd = SVD(n_factors=gs.best_params['rmse']['n_factors'],
          n_epochs=gs.best_params['rmse']['n_epochs'],
          lr_all=gs.best_params['rmse']['lr_all'],
          reg_all=gs.best_params['rmse']['reg_all'])
trainset = data.build_full_trainset()
svd.fit(trainset)

# Get predictions for the testset
predictions = svd.test(testset)

# Extract the actual ratings and predicted ratings
actual_ratings = [pred.r_ui for pred in predictions]
predicted_ratings = [pred.est for pred in predictions]

# Plot the scatter plot of actual ratings vs. predicted ratings
plt.scatter(actual_ratings, predicted_ratings, alpha=0.5)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('SVD Performance: Actual Ratings VS. Predicted Ratings')
plt.show()




# Make a prediction for user 1 and movie 302 with a rating of 3
prediction = svd.predict(1, 3671, 3)
print(prediction)


import pickle

# save the model
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd, f)