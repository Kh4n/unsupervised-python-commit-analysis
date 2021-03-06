import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import h5py
import datagen as dg
import time
import os


def sampling(args):
    # modified from keras website
    z_mean, z_log_var = args
    # by default, random_normal has mean = 0 and std = 1.0. same for tf.random.normal
    # epsilon = K.random_normal(shape=(batch, dim))
    epsilon = tf.random.normal(tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


batch_size = 1024
embed_split_index = 6
data = h5py.File("django_U0.h5", 'r')["diff_features"][:]


embed_dims = {"paths": 5000, "folders": 1000, "file_types": 10}
embed_dims_cols = [
    embed_dims["paths"], embed_dims["paths"],
    embed_dims["folders"], embed_dims["folders"],
    embed_dims["file_types"]
]
gen, vgen = dg.DiffAutencoderBasic.create_data_generators(
    data, embed_split_index, embed_dims_cols,
    batch_size=batch_size, val_split=0.2
)

tot_cols = data.shape[1]

in_data = layers.Input(shape=[tot_cols])

old_paths_categorical = layers.Lambda(lambda x: x[:, 0])(in_data)
new_paths_categorical = layers.Lambda(lambda x: x[:, 1])(in_data)
paths_embedding = layers.Embedding(embed_dims["paths"], int(embed_dims["paths"]**(1/4)), input_length=1)

old_path = paths_embedding(old_paths_categorical)
new_path = paths_embedding(new_paths_categorical)

old_folders_categorical = layers.Lambda(lambda x: x[:, 2])(in_data)
new_folders_categorical = layers.Lambda(lambda x: x[:, 3])(in_data)
folders_embedding = layers.Embedding(embed_dims["folders"], int(embed_dims["folders"]**(1/4)), input_length=1)

old_folder = folders_embedding(old_folders_categorical)
new_folder = folders_embedding(new_folders_categorical)

file_types_categorical = layers.Lambda(lambda x: x[:, 4])(in_data)
file_type = layers.Embedding(embed_dims["file_types"], int(embed_dims["file_types"]/2), input_length=1)(file_types_categorical)

is_new = layers.Lambda(lambda x: tf.expand_dims(x[:, 5], axis=-1))(in_data)

continous = layers.Lambda(lambda x: x[:, embed_split_index:])(in_data)

complete = layers.Concatenate()([old_path, new_path, old_folder, new_folder, file_type, is_new, continous])

dense = layers.Dense(32, activation="relu")(complete)
dense = layers.Dense(16, activation="relu")(dense)
# dense = layers.Dense(32, activation="relu")(dense)

out = layers.Dense(tot_cols-1)(dense)
# out = layers.Dense(tot_cols)(inter_out)

model = tf.keras.Model(inputs=[in_data], outputs=[out])
print(model.summary())


sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    loss="mse",
    optimizer=adam,
)

history = model.fit_generator(
    gen, epochs=50, validation_data=vgen, shuffle=True,
    # callbacks=[tf.keras.callbacks.TensorBoard(log_dir=logdir_name)], workers=8, use_multiprocessing=True
)

model.save(f"django_autoencoder_{int(time.time())}.h5")