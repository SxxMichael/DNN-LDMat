import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from keras.optimizers.schedules import ExponentialDecay

# Prepare Data
train_data = open('apoe_LDMat_train.txt', 'r').read()
test_data = open('apoe_LDMat_test.txt', 'r').read()
train_response = open('hip_res_train.txt', 'r').read()
test_response = open('hip_res_test.txt', 'r').read()

train_data = train_data.split("\n") #inputs
test_data = test_data.split("\n") 
train_response = train_response.split("\n") #expected outputs
test_response = test_response.split("\n") 

train_data = train_data[1:len(train_data)-1]		#Remove the 1st line and last blank line
test_data = test_data[1:len(test_data)-1]
train_response = train_response[1:len(train_response)-1]
test_response = test_response[1:len(test_response)-1]


train_in = []		# Create arrays to hold data
test_in = []
train_out = []
test_out = []

temp = []
for i in train_data:
    cur = (i.split()) #Create an array to hold data in the case
    cur = cur[1:]     #Remove the "case #" at the start of every case
    for i in range(len(cur)):   #Seperate the data in that case
        cur[i] = float(cur[i])  #Convert the String into numbers
    temp.append(np.array(cur))  #Add the array to the data set
train_in.append(temp)

temp = []		# Repeat the process for test data
for i in test_data:
    cur = (i.split())
    cur = cur[1:]
    for i in range(len(cur)):
        cur[i] = float(cur[i])
    temp.append(np.array(cur))
test_in.append(temp)

list_sum = 0
list_long = 0
list_avg = 0

for i in train_response:
    cur = i.split()
    if cur[1] != 'NA':
        list_sum = list_sum + float(cur[1])
        list_long = list_long + 1

list_avg = list_sum / list_long

temp = []
for i in train_response:
    cur = i.split()
    if cur[1] == 'NA':
        #print(cur[1])
        temp.append(float(list_avg))
    else:
        temp.append(float(cur[1]))

train_out.append(temp)

list_sum = 0
list_long = 0
list_avg = 0

for i in test_response:
    cur = i.split()
    if cur[1] != 'NA':
        list_sum = list_sum + float(cur[1])
        list_long = list_long + 1

list_avg = list_sum / list_long

temp = []
for i in test_response:
    cur = i.split()
    if cur[1] == 'NA':
        #print(cur[1])
        temp.append(float(list_avg))
    else:
        temp.append(float(cur[1]))

test_out.append(temp)  

X_train=np.squeeze(np.array(train_in)) # LDMatrix_Train
X_test=np.squeeze(np.array(test_in))   # LDMatrix_test
y_train=np.array(train_out)   #train_response
y_test=np.array(test_out)   #test_response


# Learning Rate Scheduler
iinitial_learning_rate = 0.001
decay_steps = 100000
decay_rate = 0.96
staircase = True

lr_scheduler = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=staircase)

optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler)
activation = "relu"
X_train_uptri = X_train[np.triu_indices_from(X_train, k=0)]
X_train_ld = X_train_uptri.reshape(1,-1)
d_train = X_train_ld.shape[1]
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
    Dense(y_train.shape[1])
    
])

# Model compilation
model_dnn.compile(loss="mse", optimizer=optimizer)

# Model fitting
model_dnn.fit(X_train_ld, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

# Evaluate training, test loss
evaluate_train=model_dnn.evaluate(X_train_ld, y_train)
print(evaluate_train)

# Prediction on Test LD Matrix
X_test_uptri = X_test[np.triu_indices_from(X_test, k=1)]
X_test_ld = X_test_uptri.reshape(1,-1)

# Model Evaluation
repetitions = 2000
test_err = np.zeros(repetitions)

for i in range(repetitions):
    # prediction based on test data
    predicted_test = np.squeeze(model_dnn.predict(X_test_ld, verbose=0))

    # sample individuals from the outputs
    sampled = np.random.choice(predicted_test, size=y_test.shape[1], replace=True)

    # calculate mse
    test_err[i] = np.mean((y_test - sampled) ** 2)

evaluate_test=np.mean(test_err)
print(evaluate_test)

