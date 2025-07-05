import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from keras.optimizers.schedules import ExponentialDecay

def read_file(fileName):
    temp = open(fileName, 'r').read()
    temp = temp.split("\n")
    temp = temp[:len(temp) - 1]
    return temp

X_train = read_file('apoe_train_data.txt')
X_test = read_file('apoe_test_data.txt')

y_train = read_file('hip_res_train.txt')[1:]
y_test = read_file('hip_res_test.txt')[1:]

train1 = []
targets1 = []

for i in range(len(X_train)):

    if y_train[i].split()[1:][0] == 'NA': # If NA, skip this line
        continue
    
    cur_train = X_train[i].split()
    for j in range(len(cur_train)):
        cur_train[j] = float(cur_train[j])

    cur_target = y_train[i].split()
    cur_target = [float(cur_target[1])]
    
    train1.append(np.array(cur_train))
    targets1.append(np.array(cur_target))


train2 = []
targets2 = []

for i in range(len(X_test)):

    if y_test[i].split()[1:][0] == 'NA': # If NA, skip this line
        continue
    
    cur_train = X_test[i].split()
    for j in range(len(cur_train)):
        cur_train[j] = float(cur_train[j])

    cur_target = y_test[i].split()
    cur_target = [float(cur_target[1])]
    
    train2.append(np.array(cur_train))
    targets2.append(np.array(cur_target))


train_data = np.array(train1)
test_data = np.array(train2)
X_train = train_data.reshape((train_data.shape[0], train_data.shape[1], 1))
X_test = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))

train_targets = np.array(targets1)
test_targets = np.array(targets2)
y_train = np.array(train_targets).reshape(-1)
y_test = np.array(test_targets).reshape(-1)

timesteps = X_train.shape[1]


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
epochs = 10
batch_size = 256


# Model construction
model_lstm = keras.Sequential([
    Input(shape=(timesteps, 1)),
    LSTM(10, return_sequences=True),
    LSTM(10, return_sequences=True),
    LSTM(10, return_sequences=True),
    LSTM(10, return_sequences=True),
    LSTM(10),
    Dense(1)
    
])

# Model compilation
model_lstm.compile(loss="mse", optimizer=optimizer)

# Model fitting
model_lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

# Evaluate training, test loss
evaluate_train=model_lstm.evaluate(X_train, y_train)
evaluate_test=model_lstm.evaluate(X_test, y_test)

print(evaluate_train)
print(evaluate_test)

