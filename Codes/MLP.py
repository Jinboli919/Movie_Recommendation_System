"""
Matrix Factorization
"""

"""
MLP Approach
"""


data = pd.merge(ratings, movies_m, on='movieId')

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

# add batch normalization layer
bn = layers.BatchNormalization()(concat)

# add dense layers for the neural network
dense1 = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(bn)
dense2 = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(dense1)
dense3 = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(dense2)
dense4 = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(dense3)

# add the output layer
output_layer = layers.Dense(1)(dense4)

# create the model
model = keras.Model(inputs=[user_input, item_input], outputs=output_layer)
model.compile(optimizer='sgd', loss='mean_absolute_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# split the data into training and testing datasets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# train the model
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
history = model.fit(x=[train_data['userId'], train_data['movieId']], y=train_data['rating'], epochs=15, batch_size=128, validation_data=([test_data['userId'], test_data['movieId']], test_data['rating']), callbacks=[reduce_lr])

import matplotlib.pyplot as plt

# plot the training and testing loss and metrics
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.plot(history.history['root_mean_squared_error'], label='train_rmse')
plt.plot(history.history['val_root_mean_squared_error'], label='test_rmse')
plt.title('MLP Performance')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.show()

# evaluate the model on the test data
results = model.evaluate([test_data['userId'], test_data['movieId']], test_data['rating'])
print(f'Test MAE: {results[0]}, Test RMSE: {results[1]}')


# get the predicted ratings
predictions = model.predict([test_data['userId'], test_data['movieId']])

# create a scatter plot of predicted vs actual ratings
plt.scatter(x=test_data['rating'], y=predictions, alpha=0.5)
plt.title('MLP Performance: Actual Ratings VS. Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()