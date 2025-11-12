## Deep Learning Models for Genetic Linkage Disequilibrium Matrices
This repository provides implementations of deep learning models for predicting quantitative phenotypes using linkage disequilibrium (LD) matrices as input data.
It includes multiple architectures — DNN, CNN, LSTM/BiLSTM, and Transformer — implemented in TensorFlow/Keras.

## Overview
Traditional GWAS summary statistics rely on marginal SNP effects and often ignore correlations between SNPs.
By contrast, LD matrices capture pairwise correlations among SNPs, providing richer genetic information.
This repository demonstrates how deep neural architectures can leverage LD matrices to improve predictive performance for continuous phenotypic traits (e.g., brain imaging biomarkers).

## Deep Neural Network Application
The following python code summarizes the main idea on how to construct a deep neural network with inputs being LD matrices. Detailed versions of the codes can be found in the DNN directory.
```python
# STEP 1: If the Size of a LD matrix is large, use diagonal submatrices as inputs. In the code X_train is the LD matrix
diagonal_size = 193
temp = [
    X_train[i : i + diagonal_size, i : i + diagonal_size]
    for i in range(0, len(X_train), diagonal_size)
]

X_train = np.array(temp) 

# STEP 2: Use the upper triangular or lower triangular of the sub LD matrices as the input.
triu_idx = np.triu_indices(diagonal_size, k=0) 
X_train_upper_triangles = X_train[:, triu_idx[0], triu_idx[1]]
X_train_ld = X_train_upper_triangles.reshape(1, -1)

# STEP 3: Build a DNN model
activation = "relu"
d_train = X_train_ld.shape[1]
loci = 77

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
```

## Convolutional Neural Network Application
The following python code summarizes the main idea on how to construct a convolutional neural network with inputs being LD matrices. Detailed versions of the codes can be found in the CNN directory.
```python
# STEP 1: If the Size of a LD matrix is large, use diagonal submatrices as inputs. In the code X_train is the LD matrix
diagonal_size = 193
temp = [
    train_input[i : i + diagonal_size, i : i + diagonal_size]
    for i in range(0, len(train_input), diagonal_size)
]

train_input = np.array(temp) 

# STEP 2: Build a CNN model
drop_rate=0.1
train_input = np.expand_dims(train_input, axis=-1)  # Adds the channel dimension

model_cnn1 = keras.Sequential()
model_cnn1.add(Conv2D(filters = 50, kernel_size = (50, 50), padding='same', activation= "relu", input_shape= train_input.shape[1:]))
model_cnn1.add(Dropout(drop_rate))
model_cnn1.add(layers.Flatten())
model_cnn1.add(layers.Dense(50, activation = "relu"))
model_cnn1.add(Dropout(drop_rate))
model_cnn1.add(layers.Dense(len(train_output[0])))
```

## Long Short Term Memory
The following python code summarizes the main idea on how to construct a long short term memory with inputs being LD matrices. The code for bidirectional long short term memory is similar and the detailed versions of the codes can be found in the LSTM/BiLSTM directory.
```python
# Model construction. Here X_train is the LD matrix. Different from CNN and DNN, it is not necessary to extract the sub LD matrices as inputs.
model_lstm = keras.Sequential([
    Input(shape = (X_train.shape[1], X_train.shape[2])),
    LSTM(10, return_sequences=True),
    LSTM(10, return_sequences=True),
    LSTM(10, return_sequences=True),
    LSTM(10, return_sequences=True),
    LSTM(10),
    Dense(y_train.shape[1])
    
])
```

## Transformer
The following python code summarizes the main idea on how to construct a transformer encoder with inputs being LD matrices. Detailed versions of the codes can be found in the Transformer directory.
```python
# STEP 1: If the Size of a LD matrix is large, use diagonal submatrices as inputs. In the code X_train is the LD matrix
diagonal_size = 193
temp = [
    LDMat_train_array[i : i + diagonal_size, i : i + diagonal_size]
    for i in range(0, len(LDMat_train_array), diagonal_size)
]

train_input = np.array(temp)[..., np.newaxis].astype(np.float32)

# STEP 2: Build the transformer encoder
def build_image_sequence_transformer(
    num_images=50,
    image_height=200,
    image_width=200,
    channels=1,
    embed_dim=128,
    num_heads=6,
    ff_dim=128,
    num_blocks=2,
    output_dim=100,
):
    inputs = Input(
        shape=(num_images, image_height, image_width, channels), name="input_images"
    )

    x = layers.TimeDistributed(layers.Flatten(), name="flatten")(inputs)
    x = layers.TimeDistributed(layers.Dense(embed_dim), name="linear_projection")(
        x
    ) 

    # Positional Embedding
    positions = tf.range(start=0, limit=num_images, delta=1)
    pos_embedding = layers.Embedding(
        input_dim=num_images, output_dim=embed_dim, name="pos_embedding"
    )(positions)
    pos_embedding = tf.expand_dims(pos_embedding, axis=0)
    x = layers.Lambda(lambda t: t + pos_embedding, name="add_pos_embedding")(x)

    # Transformer encoder blocks
    for i in range(num_blocks):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, name=f"mha_{i}"
        )(x, x)
        attn_output = layers.Dropout(0.1)(attn_output)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Feedforward network
        ffn = layers.Dense(ff_dim, activation="relu")(x)
        ffn = layers.Dense(embed_dim)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Pool over sequence and output regression
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(output_dim, activation="linear", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ImageSequenceTransformer")
    return model


model = build_image_sequence_transformer(
    num_images=train_input.shape[0],
    image_height=train_input.shape[1],
    image_width=train_input.shape[2],
    channels=1,
    embed_dim=128,
    num_heads=6,
    ff_dim=128,
    num_blocks=2,
    output_dim=res_train_array.shape[0],
)

```