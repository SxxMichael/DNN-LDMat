import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import keras
import math
import random

train_data = open('apoe_train_data.txt', 'r').read()
test_data = open('apoe_test_data.txt', 'r').read()
train_response = open('hip_res_train.txt', 'r').read()
test_response = open('hip_res_test.txt', 'r').read()

train_data = train_data.split("\n")
test_data = test_data.split("\n") 
train_response = train_response.split("\n")
test_response = test_response.split("\n") 

#Remove last blank line
train_data = train_data[0:len(train_data)-1]
test_data = test_data[0:len(test_data)-1]
train_response = train_response[1:len(train_response)-1]
test_response = test_response[1:len(test_response)-1]

#Create arrays to hold data
train_in = []
test_in = []
train_out = []
test_out = []

temp = []
for i in train_data:
    cur = (i.split())#Create an array to hold data in the case
    cur = cur[0:] #Remove the "case #" at the start of every case
    for i in range(len(cur)):#Seperate the data in that case
        cur[i] = float(cur[i])#Convert the String into numbers
    temp.append(np.array(cur))#Add the array to the data set
train_in.append(temp)

#Repeat process with the test data
temp = []
for i in test_data:
    cur = (i.split())
    cur = cur[0:]
    for i in range(len(cur)):
        cur[i] = float(cur[i])
    temp.append(np.array(cur))
test_in.append(temp)

# Load response data
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

#Verify the array
train_input=np.array(train_in) 
test_input=np.array(test_in)   
train_output=np.array(train_out) 
test_output=np.array(test_out) 

# Parameters
n_samples = 624
timesteps = 168
features = 1
n_classes = 32

# Load data
X = train_input
y = train_output

# Reshape data for Conv2D: (samples, height, width, channels)
X = X.reshape((n_samples, timesteps, features, 1))
y = y.reshape(n_samples)

X_train = X
y_train = y
X_test = test_input.reshape((156, timesteps, features, 1))
y_test = test_output.reshape(156)

# Build the CNN model
def build_cnn_model(input_shape, n_classes):
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Convolutional layers
        layers.Conv2D(32, (5, 1), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 1)),
        
        layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 1)),
        
        layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 1)),

        layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 1)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        #len(train_output[0])
    ])
    return model

# Create and compile the model
input_shape = (timesteps, features, 1)
model = build_cnn_model(input_shape, n_classes)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.1)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train,
                   epochs=100,
                   batch_size=32,
                   validation_data=(X_test, y_test),
                   verbose=1)

# Evaluate the model
print("Training loss", model.evaluate(X_test, y_test, verbose = 0))

count = 0
numTimes = 10
evaluate_test=0

output = model(X_test)
positions = random.sample(range(0, len(test_output[0])), len(test_output[0])) #get the data from random positions to compare with

while (count < numTimes):
    count = count + 1

    sum = 0
    for i in range(len(test_output[0])):
        sum+= math.pow((output[positions[i]].numpy().item()-test_output[0][i]),2) #summing differences squared
    
    evaluate_test_tmp = sum/len(test_output[0]) #average of differences
    evaluate_test= evaluate_test + evaluate_test_tmp

evaluate_test=evaluate_test/numTimes

print('Test loss', evaluate_test)
