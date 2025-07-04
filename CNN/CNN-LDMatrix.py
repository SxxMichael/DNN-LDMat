import numpy as np
import tensorflow as tf
import keras
import math
from keras import layers

import numpy as np

def load_ld_matrix(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Skip the first row (SNP names)
    data = []
    for line in lines[1:]:  # Start from the second row
        values = line.split()[1:]  # Skip case number
        try:
            values = list(map(float, values))  # Convert all elements to float
            data.append(values)
        except ValueError:
            print(f"Skipping malformed line: {line.strip()}")  # Debugging in case of bad lines

    # Find max row length to ensure uniform shape
    max_len = max(len(row) for row in data)
    if any(len(row) != max_len for row in data):
        print("Failed somewhere: row length mismatch")
    data_padded = [row + [0.0] * (max_len - len(row)) for row in data]  # Pad shorter rows with zeros

    return np.array(data_padded, dtype=np.float32)

def load_tests(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines[1:]:  # Start from the second row
        values = line.split()[1]# Skip case number, only second case num
        try:
            values = float(values)  # Directly convert to a float
            data.append(values)  # Append to the list
        except ValueError:
            print(values)
            print(f"Skipping malformed line: {line.strip()}")  # Debugging in case of bad lines
    
    return np.array(data, dtype=np.float32)

print("Case 3")


    
# Load datasets
train_input = load_ld_matrix("LDMatrix_train.txt")
test_input = load_ld_matrix("LDMatrix_test.txt")

print(f"Train Input Shape: {train_input.shape}")
print(f"Test Input Shape: {test_input.shape}")

# Get diagonal of train_input
diagonal_size = 193
temp = [
    train_input[i : i + diagonal_size, i : i + diagonal_size]
    for i in range(0, len(train_input), diagonal_size)
]

train_input = np.array(temp) 

# Get diagonal of test_input
temp = [
    test_input[i : i + diagonal_size, i : i + diagonal_size]
    for i in range(0, len(test_input), diagonal_size)
]

test_input = np.array(temp)

print(train_input.shape)
print(test_input.shape)

# Duplicating train_output to match # of input cases
train_output =[]
temp = load_tests("response_train.txt")
#temp = np.expand_dims(train_output, axis=-1)  # Adds a second dimension, the ,1 at the end
train_output = np.tile(temp, (len(train_input),1))
print(f"Test Output Shape: {train_output.shape}")

# Duplicating test_output to match # of input cases
test_output =[]
temp = load_tests("response_test.txt")
#temp = np.expand_dims(train_output, axis=-1)  # Adds a second dimension, the ,1 at the end
test_output = np.tile(temp, (len(test_input),1))
print(f"Test Output Shape: {test_output.shape}")


# Define exponential decay schedule, copy pasted from previou lessons
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


print(f"train_input shape: {train_input.shape}")
print(f"train_out shape: {train_output.shape}")
print(f"train_input shape: {test_input.shape}")
print(f"train_out shape: {test_output.shape}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

drop_rate=0.1

train_input = np.expand_dims(train_input, axis=-1)  # Adds the channel dimension
test_input = np.expand_dims(train_input, axis=-1)  # Adds the channel dimension

model_cnn1 = keras.Sequential()
#Plug in the number of filters, size of filters, activation functions, and the 
model_cnn1.add(Conv2D(filters = 50, kernel_size = (50, 50), padding='same', activation= "relu", input_shape= train_input.shape[1:]))
model_cnn1.add(Dropout(drop_rate))


# Flatten the output of the Conv1D layer
model_cnn1.add(layers.Flatten())



model_cnn1.add(layers.Dense(50, activation = "relu"))
model_cnn1.add(Dropout(drop_rate))


# not sure if we want the final to have a activation
model_cnn1.add(layers.Dense(len(train_output[0])))



# Summary of your model
model_cnn1.summary()


optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
loss = 'mse'


# Model compilation
model_cnn1.compile(optimizer = optimizer, loss = loss)

batch_size = 32
epochs = 200
#May need to change or manually set epochs/batch size as needed
model_cnn1.fit(train_input, train_output, batch_size = batch_size, epochs = epochs, verbose = 0)

output = np.array(model_cnn1.predict(test_input))
print(output.shape)
#print(output)


evaluate_test = model_cnn1.evaluate(train_input, train_output, verbose = 0)
    
print('Train loss', evaluate_test)


# Model Evaluation
import random
import math
avg = 0
repetitions = 2000
for _ in range(repetitions):
    #repeat this 2000 times.focus on with replacement.
    #positions = random.sample(range(0, output.shape[1]), test_output.shape[1])#get 219 numbers between 0 and 873, no need for positions if replacing
    #print(positions)
    # Initialize the evaluation metric
    #without_r = 0
    with_r = 0
    
    # Iterate through all test cases in output
    for test in test_output:  # Each "test" corresponds to a diagonal square
        #temp_without = 0
        temp_with = 0
    
        
        # Iterate through all indices in test_out
        for i in range(test_output.shape[1]):  # Assuming test_out.shape[1] = 219
            # Calculate the squared difference
            #without_val = test[positions[i]].item()
            with_val = test[ int(random.random()*test_output.shape[1]) ].item()
            
            #diff_without = without_val - test_out[0][i]
            diff_with = with_val - test_output[0][i]
            
            #temp_without += math.pow(diff_without, 2)
            temp_with += math.pow(diff_with, 2)
    
        # Average the MSE for this test case
        #temp_without /= test_out.shape[1]
        temp_with /= test_output.shape[1]
        
        #without_r += temp_without
        with_r += temp_with
    
    # Average the MSE across all test cases
    #without_r /= len(output)
    with_r /= len(output)

    avg +=with_r
# Print the result
print('With Replacement Test loss', avg/repetitions) #pretty sure

