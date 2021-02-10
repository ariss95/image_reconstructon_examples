import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Linear(784, 64, True)

        self.decoder = nn.Linear(64, 784, True)
    def forward(self, x):
        h = torch.sigmoid(self.encoder(x))

        x = torch.sigmoid(self.decoder(h))

        return x