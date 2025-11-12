# load libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input,
    Dense,
    LayerNormalization,
    GlobalAveragePooling1D,
    Lambda,
    Dropout,
    MultiHeadAttention,
    Layer,
)
from tensorflow.keras.models import Model

# read data
LDMat_train = pd.read_csv("apoe_LDMat_train.txt", sep=" ")
LDMat_test = pd.read_csv("apoe_LDMat_test.txt", sep=" ")
res_train = pd.read_csv("hip_res_train.txt", sep=" ")
res_test = pd.read_csv("hip_res_test.txt", sep=" ")

# Fill NA by means
res_train = res_train.fillna(res_train.mean(numeric_only=True))
res_test = res_test.fillna(res_test.mean(numeric_only=True))

# transfer to pd dataframe to np array
LDMat_train_array = LDMat_train.to_numpy()
LDMat_test_array = LDMat_test.to_numpy()
res_train_array = res_train.to_numpy()
res_test_array = res_test.to_numpy()

# Get diagonal of train_input
diagonal_size = 168
temp = [
    LDMat_train_array[i : i + diagonal_size, i : i + diagonal_size]
    for i in range(0, len(LDMat_train_array), diagonal_size)
]

train_input = np.array(temp)[..., np.newaxis].astype(np.float32)

# Get diagonal of test_input
temp = [
    LDMat_test_array[i : i + diagonal_size, i : i + diagonal_size]
    for i in range(0, len(LDMat_test_array), diagonal_size)
]

test_input = np.array(temp)[..., np.newaxis].astype(np.float32)


def build_image_sequence_transformer(
    num_images=50,
    image_height=200,
    image_width=200,
    channels=1,
    embed_dim=128,
    num_heads=4,
    ff_dim=256,
    num_blocks=2,
    output_dim=100,
):
    # Input: (batch, num_images, H, W, C)
    inputs = Input(
        shape=(num_images, image_height, image_width, channels), name="input_images"
    )

    # Step 1: Flatten and linearly project each image
    x = layers.TimeDistributed(layers.Flatten(), name="flatten")(inputs)
    x = layers.TimeDistributed(layers.Dense(embed_dim), name="linear_projection")(
        x
    )  # (batch, num_images, embed_dim)

    # Step 2: Add learnable positional embeddings
    positions = tf.range(start=0, limit=num_images, delta=1)
    pos_embedding = layers.Embedding(
        input_dim=num_images, output_dim=embed_dim, name="pos_embedding"
    )(positions)
    pos_embedding = tf.expand_dims(pos_embedding, axis=0)  # (1, num_images, embed_dim)
    x = layers.Lambda(lambda t: t + pos_embedding, name="add_pos_embedding")(x)

    # Step 3: Transformer encoder blocks
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

    # Step 4: Pool over sequence and output regression
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
    num_heads=2,
    ff_dim=128,
    num_blocks=2,
    output_dim=res_train_array.shape[0],
)

# === Compile ===
model.compile(optimizer="adam", loss="mse")

# === Training Model ===
train_input_ld = np.expand_dims(train_input, axis=0)
res_train_array_ld = np.expand_dims(res_train_array, axis=0)
res_train_array_ld = res_train_array_ld.reshape(1, res_train_array.shape[0])
model.fit(train_input_ld, res_train_array_ld, batch_size=8, epochs=10, verbose = 0)
train_err = model.evaluate(train_input_ld, res_train_array_ld, verbose=0, batch_size=8)

# Model Evaluation
repetitions = 2000
test_err = np.zeros(repetitions)

test_input_ld = np.expand_dims(test_input, axis=0)
res_test_array_ld = np.expand_dims(res_test_array, axis=0)
res_test_array_ld = res_test_array_ld.reshape(1, res_test_array.shape[0])

for i in range(repetitions):
    # prediction based on test data
    predicted_test = np.squeeze(model.predict(test_input_ld, verbose=0, batch_size=8))

    # sample individuals from the outputs
    sampled = np.random.choice(
        predicted_test, size=res_test_array_ld.shape[1], replace=True
    )

    # calculate mse
    test_err[i] = np.mean((res_test_array_ld - sampled) ** 2)

test_err = np.mean(test_err)
print(train_err, test_err)
