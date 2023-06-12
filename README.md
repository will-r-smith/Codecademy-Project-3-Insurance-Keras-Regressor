# Codecadedmy Portfolio Project 3: Medical Insurance Modelling

*This project was completed as a part of the * **Build Deep Learning Models with TensorFlow** *skill path curriculum on Codecademy. The purpose of the project was to practice the using Tensorflow neural networks to perform modelling on a simple dataset before applying the knowledge on a larger project.*

### Description

The `scripts/insurance.py` file performs an iterative RandomizedSearchCV() for a KerasRegressor model. The hyperparameters varied in the tuning process were the batch size and learning rate. In conjunction with this, the effect of varying the design of the model itself was investigated. The model attributes that were varied were: the number of hidden layers, the activation functions, the neurons per layer and the dropout rates.

1000 iterations were performed and the top 20 models were inspected. From these, a final model was obtained which included a single hidden layer with 64 neurons, no dropout, and an sigmoid activation function. A batch size of 8 and a learning rate of 0.08 were used to obtain the final model and predicitons.

The model's loss convergence and predictions can be found in the outputs folder.