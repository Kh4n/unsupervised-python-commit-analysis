import tensorflow as tf
import datagen as dg
import numpy as np
import h5py
import os

embed_split_index = 6
embed_dims = {"paths": 5000, "folders": 1000, "file_types": 10}
embed_dims_cols = [
    embed_dims["paths"], embed_dims["paths"],
    embed_dims["folders"], embed_dims["folders"],
    embed_dims["file_types"]
]

data_test = h5py.File("django_U0_test2.h5", 'r')["diff_features"][:]
tgen,_ = dg.DiffAutencoderBasic.create_data_generators(
    data_test, embed_split_index, embed_dims_cols,
    batch_size=1, val_split=0.0, shuffle=False
)

model = tf.keras.models.load_model("django_autoencoder_larg2.h5")
print(model.summary())

with open("OUT", "w") as f:
    with np.printoptions(threshold=np.inf):
        for i in range(len(tgen)):
            f.write( str(model.test_on_batch(tgen[i][0], tgen[i][1])) )
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())