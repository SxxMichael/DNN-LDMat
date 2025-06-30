import pandas as pd
import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from keras.optimizers.schedules import ExponentialDecay

def load_train(file):
    # Skip first row and first column
    df = pd.read_csv(file, sep='\s+', header=None, skiprows=1)
    df = df.iloc[:, 1:]
    return df.to_numpy(dtype=np.float64)

def load_targets(file):
    # Skip first row and first column
    df = pd.read_csv(file, sep='\s+', header=None, skiprows=1)
    df = df.iloc[:, 1] # Take second column only
    return df.to_numpy(dtype=np.float64)

y_train = load_targets("response_train.txt")
y_test = load_targets("response_test.txt")

X_train = load_train("LDMatrix_train.txt")[:y_train.shape[0]]
X_test = load_train("LDMatrix_test.txt")[:y_test.shape[0]]


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
