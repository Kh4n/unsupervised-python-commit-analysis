import tensorflow as tf
import numpy as np

class DiffAutencoderBasic(tf.keras.utils.Sequence):
    def __init__(self, data, minmaxes, embed_split_index, batch_size=32):
        self.data = data
        self.minmaxes = minmaxes
        self.ESI = embed_split_index
        self.batch_size = batch_size
        self.mins = self.minmaxes[:,0]
        self.ranges = self.minmaxes[:,1] - self.minmaxes[:,0]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.data)//self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        embed_in = self.data[index*self.batch_size:(index+1)*self.batch_size, :self.ESI]
        continuous = (self.data[index*self.batch_size:(index+1)*self.batch_size, self.ESI:] - self.mins[self.ESI:])/self.ranges[self.ESI:]

        y = (self.data[index*self.batch_size:(index+1)*self.batch_size] - self.mins)/self.ranges
        
        return np.concatenate([embed_in, continuous], axis=-1), y
    
    @classmethod
    def create_data_generators(cls, data, embed_split_index, embed_dims_cols, batch_size=32, val_split=0.2, shuffle=True):
        for i in range(len(embed_dims_cols)):
            a = data[:, i]
            a[a >= embed_dims_cols[i]] = 0
        
        if shuffle:
            np.random.shuffle(data)
        train_split = len(data) - int(val_split*len(data))

        minmaxes = []
        for i in range(data.shape[1]):
            minmaxes.append([min(data[:, i]), max(data[:, i])])

        minmaxes = np.asfarray(minmaxes)
        train_gen = cls(data[0:train_split], minmaxes, embed_split_index, batch_size)
        val_gen = cls(data[train_split:], minmaxes, embed_split_index, batch_size)

        return (train_gen, val_gen)