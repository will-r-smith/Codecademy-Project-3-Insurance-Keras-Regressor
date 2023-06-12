from keras.layers import Dense, Dropout, InputLayer
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations_with_replacement

# create features and labels dataframes
data = pd.read_csv('data/insurance.csv')
features = data.drop(['charges'], axis=1)
features = pd.get_dummies(features) # one hot encode the features dataframe
labels = data['charges']

# perform a test train split on the data with 20% of the data as test data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough') # standardize the data
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# create a neural network regressor model using Keras Sequential class 
def create_model(loss='mean_squared_error', metrics=['mse'], dropout_rate=[0.0], activation=['relu'], neurons=[16], hidden_layers=1, learning_rate=0.01):
    model = Sequential(name = "pollution_neural_network") # initialize a sequential model
    input = InputLayer(input_shape=(X_train.shape[1],)) # create input layer
    model.add(input) # add the input layer to the model
    for l in range(hidden_layers):
        model.add(Dense(neurons[l], activation=activation[l])) # add a hidden layer
        model.add(Dropout(round(dropout_rate[l], 3))) # add a dropout layer
    model.add(Dense(1, activation='linear')) # add the output layer
    optimizer = Adam(lr=learning_rate) # define the optimizer
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics) # compile the model
    return model

# create a keras regressor with the create_model function
regressor = KerasRegressor(build_fn=create_model, verbose=1)

# define parameters for the random search
loss = ['mean_squared_error']
metrics = ['mae']
hidden_layers = [1, 2, 3, 4, 5]
available_activation_funcs = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
dropout_range = [0, 0.3]
neuron_options = [32, 64, 128, 256]
learning_rate = [0.01, 0.05]
epochs = 600

# create the parameter grid
param_grid = dict(
    batch_size=np.arange(3, 16, 1),  # Adjust the step size as per your preference
    epochs=[epochs],  # Adjust the step size as per your preference
    activation=list(combinations_with_replacement(available_activation_funcs, hidden_layers[-1])),
    neurons=list(combinations_with_replacement(neuron_options, hidden_layers[-1])),
    hidden_layers=hidden_layers,
    dropout_rate=[[random.uniform(dropout_range[0], dropout_range[1]) for _ in range(hidden_layers[-1])]],
    loss=loss,
    metrics=metrics,
    learning_rate=np.arange(0.01, 0.11, 0.01)  # Adjust the step size as per your preference
)

# perform a random search with the parameters and save the results to a csv file
n_iterations = 1000
grid = RandomizedSearchCV(estimator=regressor, param_distributions=param_grid, cv=5, verbose=1, n_iter=n_iterations) 
stop = EarlyStopping(monitor='mae', mode='min', verbose=1, patience = 50) # stop the training if the validation loss does not improve for 50 epochs
grid_result = grid.fit(X_train, y_train, callbacks = [stop]) # fit the model with the training data

# output the best model and test and score the model on the test data
best_model = grid_result.best_estimator_ # get the best model
y_pred = best_model.predict(X_test) # predict the test data
score = best_model.score(X_test, y_test) # score the model on the test data
print(score)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# create a report of the results, sorted by rank and save it to a csv file
report = pd.DataFrame(grid_result.cv_results_) 
report = report.sort_values(by=['rank_test_score'])
with open('outputs/reports/model_iterations.csv', 'a') as f:
    report.to_csv(f, header=False, index=False)
