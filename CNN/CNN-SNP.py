import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import layers

#train = %load SNP.txt
#test = %load response.txt
import math

in_train = open('SNP.txt', 'r').read()
in_test = open('response.txt', 'r').read()


in_train = in_train.split("\n")
in_test = in_test.split("\n")

in_train = in_train[1:1093]
in_test = in_test[1:1093]

train = []
test = []
for i in in_train:
    cur = (i.split())
    cur = cur[1:] # remove the "#"
    for i in range(len(cur)):
        cur[i] = float(cur[i])
    train.append(np.array(cur))


for i in in_test:
    cur = (i.split())
    cur = cur[1:]
    for i in range(len(cur)):
        cur[i] = float(cur[i])
    test.append(np.array(cur))

train_data = np.array(train[:math.floor(len(train)*0.8)])
train_targets= np.array(test[:math.floor(len(test)*0.8)])
test_data = np.array(train[math.floor(len(train)*0.8):])
test_targets = np.array(test[math.floor(len(test)*0.8):])
print("Training data shape:", train_data.shape)
print("Training targets shape:", train_targets.shape)
print("Test data shape:", test_data.shape)
print("Test targets shape:", test_targets.shape)

# Define exponential decay schedule
initial_learning_rate = 0.1
decay_steps = 1000
decay_rate = 0.98
staircase = True

from keras.optimizers.schedules import ExponentialDecay
# The learning rate schedule will set the learning rate to be 
# initial_learning_rate * decay_rate ^ (global_step / decay_steps).
# If staircase == True, the division glocal_step/decay_steps will be the integer division
lr_schedule = ExponentialDecay(
    initial_learning_rate = initial_learning_rate,
    decay_steps = decay_steps,
    decay_rate = decay_rate,
    staircase = staircase)

loss = "mse"
drop_rate=0.1

epochs = 200 
batch_size = 32

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.layers import Dropout


average = 0
ret = []
model_cnn1 = keras.Sequential()
model_cnn1.add(layers.Conv1D(filters = 50, kernel_size = 500, activation = "relu", input_shape=(8299, 1)))
model_cnn1.add(Dropout(drop_rate))
    
# Flatten the output of the Conv1D layer
model_cnn1.add(Flatten())
model_cnn1.add(layers.Dense(50, activation = "relu"))
model_cnn1.add(Dropout(drop_rate))
    
    
model_cnn1.add(layers.Dense(1))
    
# Summary of your model
model_cnn1.summary()

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
# Model compilation
model_cnn1.compile(optimizer = optimizer, loss = loss)
model_cnn1.fit(train_data, train_targets, batch_size = batch_size, epochs = epochs, verbose = 0)
    
# Model Evaluation
    
evaluate_test = model_cnn1.evaluate(test_data, test_targets, verbose = 0)
    
print("Train loss: ", model_cnn1.evaluate(train_data, train_targets, verbose = 0))
print('Test loss', evaluate_test)
