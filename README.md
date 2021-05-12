# image reconstruction
this repository contains two examples of reconstructing input images/videos with deep learning methods using pyTorch.
# Auto Encoder
A simple auto encoder as described [here](https://en.wikipedia.org/wiki/Autoencoder#Basic_architecture):

it maps the images into the code, and then maps the code to the reconstructed images.

dataset: MNIST

# RNN
a simple, stacked Reccurent Neural Network model from pyTorch with ReLU activation function.

[pytorch.org/docs/stable/generated/torch.nn.RNN.html](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)

the loss function (mean square error) is minimized using Adam optimization.

dataset: [moving MNIST](https://www.cs.toronto.edu/~nitish/unsupervised_video/)

## data_loader.py
This script contains a class for easy loading of data from the moving MNIST dataset.

The dataset is splitted in training-validation and testing subsets.

The training set is being shuffled after loading all data from it.


For every method, the properties of the model are in the file model_"method name".py and the rest under the train_"method name".py
