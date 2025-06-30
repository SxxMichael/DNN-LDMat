import numpy as np
import tensorflow as tf
import keras
import math
import random
from keras import layers

train_data = open('apoe_LDMat_train.txt', 'r').read()
test_data = open('apoe_LDMat_test.txt', 'r').read()
train_response = open('hip_res_train.txt', 'r').read()
test_response = open('hip_res_test.txt', 'r').read()

train_data = train_data.split("\n") #inputs
test_data = test_data.split("\n") 
train_response = train_response.split("\n") #expected outputs
test_response = test_response.split("\n") 

#Remove the 1st line and last blank line
train_data = train_data[1:len(train_data)-1]
test_data = test_data[1:len(test_data)-1]
train_response = train_response[1:len(train_response)-1]
test_response = test_response[1:len(test_response)-1]

#Create arrays to hold data
train_in = []
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

#Repeat process with the test data
temp = []
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

#Verify the array
train_input=np.array(train_in) # LDMatrix_Train
test_input=np.array(test_in)   # LDMatrix_test
train_output=np.array(train_out)   #train_response
test_output=np.array(test_out)   #test_response

epochs = 100
batch_size = 32

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
loss = "mse"

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model_cnn = keras.Sequential()

model_cnn.add(Conv2D(filters = 64, kernel_size = (5, 5), activation="relu", input_shape=(len(train_input[0]), len(train_input[0][0]), 1)))
model_cnn.add(MaxPooling2D(pool_size=(2, 1)))

# Flatten and add dense layer
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(len(train_output[0])))  

# Model compilation
model_cnn.compile(optimizer = optimizer, loss = loss, metrics=['accuracy'])

history = model_cnn.fit(train_input, train_output, batch_size = batch_size, epochs = epochs, validation_data=(train_input, train_output))

print("Training loss", model_cnn.evaluate(train_input, train_output, verbose = 0))

# Model Evaluation
output = model_cnn(test_input) #output produced by test data

count = 0
numTimes = 100
evaluate_test=0

while (count < numTimes):
    count = count + 1

    sum = 0
    for i in range(len(test_output[0])):
        sum+= math.pow((output[0][positions[i]].numpy().item()-test_output[0][i]),2) #summing differences squared
    
    evaluate_test_tmp = sum/len(test_output[0])
    evaluate_test= evaluate_test + evaluate_test_tmp

evaluate_test=evaluate_test/numTimes

print('Test loss', evaluate_test)