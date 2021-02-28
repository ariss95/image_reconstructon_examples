import data_loader
import torch
import torch.nn as nn

class first_RNN(nn.Module):
    def __init__(self, input):
        super(first_RNN, self).__init__()
        #TODO add layers
        self.device = torch.device('cpu')
        self.original_input = input #add size
        self.compressed_input = int(self.original_input/4) #from 64*64 to 32*32
        self.compression = nn.Linear(self.original_input, self.compressed_input)
        self.rnn = nn.RNN(input_size=self.compressed_input, hidden_size=2*self.original_input, num_layers=2, nonlinearity = "relu").to(device=self.device)
        self.output = nn.Linear(2*self.original_input, self.original_input).to(device=self.device)

    def forward(self, data):
        compressed = self.compression(data)
        out, hidden = self.rnn(compressed, None)# NONE for h0 = [0..0]
        out = self.output(out)
        return out
