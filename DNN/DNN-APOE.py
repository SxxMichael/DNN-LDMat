import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
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


X_train = np.array(train1)
y_train = np.array(targets1)
X_test = np.array(train2)
y_test = np.array(targets2)


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
activation = "relu"
d_train = X_train.shape[1]
loci = 77

# Parameters
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
evaluate_train=model_dnn.evaluate(X_train, y_train)
print('Train loss', evaluate_train)

evaluate_test=model_dnn.evaluate(X_test, y_test)
print('Test loss', evaluate_test)