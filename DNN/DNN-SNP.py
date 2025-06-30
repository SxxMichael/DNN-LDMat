import math
import pandas as pd
import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from keras.optimizers.schedules import ExponentialDecay


SNP = open('SNP.txt', 'r').read()
response = open('response.txt', 'r').read()

SNP = SNP.split("\n")[1:1093]
response = response.split("\n")[1:1093]


train = []
targets = []

for i in range(len(SNP)):
    
    cur_train = SNP[i].split()[1:]
    cur_target = response[i].split()[1:]
    
    train.append(np.array(cur_train).astype(np.float64))
    targets.append(np.array(cur_target).astype(np.float64))


# 80% used for training, 20% used for testing
cutoff = math.floor(len(train) * 0.8)

X_train = np.array(train[:cutoff])
y_train = np.array(targets[:cutoff])
X_test = np.array(train[cutoff:])
y_test = np.array(targets[cutoff:])


initial_learning_rate = 0.001
decay_steps = 100000
decay_rate = 0.96
staircase = True

# Learning rate schedule
lr_scheduler = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=staircase)

optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler)
activation = "sigmoid"
d_train = X_train.shape[1]
loci = 77


# Hyperparameters
epochs = 100
batch_size = 256


# Model construction
model_dnn = keras.Sequential([
    
    Input(shape=(d_train,)),
    Dense(3 * loci, activation=activation),
    Dropout(0.2),
    Dense(loci, activation=activation),
    Dense(22, activation=activation),
    Dropout(0.5),
    Dense(5, activation=activation),
    Dense(1)
    
])

# Model compilation
model_dnn.compile(loss="mse", optimizer=optimizer)

# Model fitting
model_dnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

# Evaluate training, test loss
evaluate_train = model_dnn.evaluate(X_train, y_train)
print('Train loss', evaluate_train)

evaluate_test = model_dnn.evaluate(X_test, y_test)
print('Test loss', evaluate_test)




