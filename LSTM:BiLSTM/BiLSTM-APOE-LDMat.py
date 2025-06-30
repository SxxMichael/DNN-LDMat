import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from keras.optimizers.schedules import ExponentialDecay

def read_file(fileName):
    temp = open(fileName, 'r').read()
    temp = temp.split("\n")
    temp = temp[:len(temp) - 1]
    return temp

X_train = read_file('apoe_LDMat_train.txt')[1:]
X_test = read_file('apoe_LDMat_test.txt')[1:]

y_train = read_file('hip_res_train.txt')[1:]
y_test = read_file('hip_res_test.txt')[1:]

train1 = []
targets1 = []

for i in range(len(X_train)):

    if y_train[i].split()[1:][0] == 'NA': # If NA, skip this line
        continue
    
    train1.append(np.array(X_train[i].split()[1:], dtype=float))
    targets1.append(np.array(y_train[i].split()[1], dtype=float))


train2 = []
targets2 = []

for i in range(len(y_test)):

    if y_test[i].split()[1:][0] == 'NA': # If NA, skip this line
        continue
    
    train2.append(np.array(X_test[i].split()[1:], dtype=float))
    targets2.append(np.array(y_test[i].split()[1], dtype=float))

X_train = np.array(train1)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(train2)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

y_train = np.array(targets1).reshape(-1)
y_test = np.array(targets2).reshape(-1)


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


# Parameters
epochs = 5
batch_size = 256


# Model construction
model_bilstm = keras.Sequential([
    
    Bidirectional(LSTM(10, return_sequences=True)),
    Bidirectional(LSTM(10)),
    Dense(1)
    
])

# Model compilation
model_bilstm.compile(loss="mse", optimizer=optimizer)

# Model fitting
model_bilstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

# Evaluate training, test loss
evaluate_train=model_bilstm.evaluate(X_train, y_train)
print('Train loss', evaluate_train)

evaluate_test=model_bilstm.evaluate(X_test, y_test)
print('Test loss', evaluate_test)

