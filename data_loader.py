import numpy as np
import matplotlib.pyplot as plt
import plot_utils
import torch


# this class is used for loading the dataset from the .npy file
class Moving_MNIST_Loader:
    def __init__(self, path, time_steps, flatten=True):
        
        self.data = np.load(path).astype('float32')
        
        if time_steps < self.data.shape[0]:
            self.data = self.data[:time_steps]
        self.num_frames, self.num_samples, self.size = self.data.shape[0], self.data.shape[1], self.data.shape[2:]

        if flatten:
            self.data = self.data.reshape([self.num_frames, self.num_samples, -1])

        self.train_set_size = int(self.num_samples * 0.8)
        self.validation_size = self.train_set_size + int(self.num_samples * 0.1)
        self.train = self.data[:, :self.train_set_size, ...]
        self.validate = self.data[:, self.train_set_size: self.validation_size, ...]
        self.test = self.data[:, self.validation_size:, ...]
        self.train_index = 0
        self.validation_index = 0
        self.testing_index = 0

        print("loading of moving MNIST completed")
    
    def shuffle(self):
        indices = np.random.permutation(self.train_set_size)
        self.train = self.train[:, indices, ...]
    
    def get_batch(self, set, batch_size):
        if set=="train":
            if self.train_index + batch_size -1>= self.train_set_size:
                self.shuffle()
                self.train_index = 0

            batch = self.train[:, self.train_index:self.train_index + batch_size, ...]
            self.train_index += batch_size
            return batch
        elif set=="test":
            batch = self.test[:, self.testing_index: self.testing_index + batch_size, ...]
            self.testing_index += batch_size
            return batch
        else:
            if self.validation_index + batch_size -1 >= self.validate.shape[1]:
                self.validation_index = 0
                return []
            batch = self.validate[:, self.validation_index: self.validation_index + batch_size, ...]
            self.validation_index += batch_size
            return batch