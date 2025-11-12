# Load Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import keras
from keras import layers, models
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    Layer,
    Reshape,
    TimeDistributed,
    Lambda,
)
from tensorflow.keras.models import Model

# Import Data
apoe_train = pd.read_csv("apoe_train_data.txt", sep=" ", header=None).to_numpy()
apoe_test = pd.read_csv("apoe_test_data.txt", sep=" ", header=None).to_numpy()


hip_train = pd.read_csv("hip_res_train.txt")
hip_train = hip_train["x"].str.split(expand=True)
hip_train.columns = ["index", "value"]
hip_train["value"] = pd.to_numeric(hip_train["value"], errors="coerce")
hip_train = hip_train.drop(columns="index")
hip_train["value"] = hip_train["value"].fillna(hip_train["value"].mean())
hip_train = hip_train.to_numpy()

hip_test = pd.read_csv("hip_res_test.txt")
hip_test = hip_test["x"].str.split(expand=True)
hip_test.columns = ["index", "value"]
hip_test["value"] = pd.to_numeric(hip_test["value"], errors="coerce")
hip_test = hip_test.drop(columns="index")
hip_test["value"] = hip_test["value"].fillna(hip_test["value"].mean())
hip_test = hip_test.to_numpy()


# Build Model
# Split into overlapping windows
def sliding_windows(X, window_size, stride):
    n_samples, seq_len, feat_dim = X.shape
    windows = []
    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        windows.append(X[:, start:end, :])
    return np.stack(
        windows, axis=1
    )  # shape: (n_samples, n_windows, window_size, feat_dim)


def build_model(n_windows, window_size, feature_dim, embed_dim, num_heads, ff_dim):
    inputs = Input(shape=(n_windows, window_size, feature_dim))

    # Create transformer layers ONCE (no custom class)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    ffn_dense1 = Dense(ff_dim, activation="relu")
    ffn_dense2 = Dense(embed_dim)
    ln1 = LayerNormalization(epsilon=1e-6)
    ln2 = LayerNormalization(epsilon=1e-6)
    dropout1 = Dropout(0.1)
    dropout2 = Dropout(0.1)

    # Define reusable transformer function
    def apply_transformer(x):
        attn_output = attn(x, x)
        attn_output = dropout1(attn_output)
        out1 = ln1(x + attn_output)

        ffn_output = ffn_dense1(out1)
        ffn_output = ffn_dense2(ffn_output)
        ffn_output = dropout2(ffn_output)
        return ln2(out1 + ffn_output)

    # Apply transformer to each window
    x = TimeDistributed(tf.keras.layers.Lambda(apply_transformer))(inputs)

    # Pool each window’s output
    x = TimeDistributed(GlobalAveragePooling1D())(x)

    # Pool across windows
    x = GlobalAveragePooling1D()(x)

    # Dense head
    x = Dense(128, activation="relu")(x)
    outputs = Dense(1, activation="linear")(x)

    return Model(inputs, outputs)


# Model Parameters
feature_dim = 1
window_size = 40
stride = 20  # overlap
X_train = apoe_train[..., np.newaxis].astype(np.float32)
X_test = apoe_test[..., np.newaxis].astype(np.float32)

X_train_windows = sliding_windows(X_train, window_size, stride)
X_test_windows = sliding_windows(X_test, window_size, stride)

n_windows = X_train_windows.shape[1]
embed_dim = feature_dim
num_heads = 6
ff_dim = 64

model = build_model(n_windows, window_size, feature_dim, embed_dim, num_heads, ff_dim)
model.compile(optimizer="adam", loss="mse")

model.fit(X_train_windows, hip_train, batch_size=8, epochs=10, verbose=0)

train_err = model.evaluate(X_train_windows, hip_train, batch_size=8, verbose=0)
test_err = model.evaluate(X_test_windows, hip_test, batch_size=8, verbose=0)
print(train_err, test_err)
