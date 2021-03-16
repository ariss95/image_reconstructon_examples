import data_loader
import torch
import torch.nn as nn

class l1_l1(nn.Module):
    def __init__(self, input):
        super(first_RNN, self).__init__()
        self.input = input
        self.compressed = int(self.input/4)
        self.device = torch.device('cpu')
        self.measurement_matrix = np.asarray(
            np.random.RandomState().uniform(
                low=-np.sqrt(6.0 / (self.compressed + self.input)),
                high=np.sqrt(6.0 / (self.compressed + self.input)),
                size=(self.compressed, self.input)
            ) / 2.0, dtype=np.float32)
        self.A = torch.tensor(self.measurement_matrix, device=self.device, dtype=torch.float32, requires_grad=True)
        
        
        self.Dict_D = torch.tensor(device=self.device, dtype=torch.float32, requires_grad=True)#############FIX
        self.h_0 = torch.zeros((self.batch_size, 2*self.input), device=self.device, dtype=self.dtype,
                               requires_grad=True)

        '''
        

    def forward(self, data):
        compressed = self.compression(data)
        out, hidden = self.rnn(compressed, None)# NONE for h0 = [0..0]
        out = self.output(out)
        return out
