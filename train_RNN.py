import data_loader as dl
import model_RNN
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import time
path = 'movingMnist/mnist_test_seq.npy'
data_loader = dl.Moving_MNIST_Loader(path=path, time_steps=20, load_only=-1,
                                      flatten=True, scale=False)
device = torch.device('cuda')
model = model_RNN.first_RNN(4096).to(device)
learning_rate = 0.001
epochs = 500
batch_size = 64
training_samples = 8000
#print(data_loader.data[0].shape)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def fit(model, dataloader):
    model.train()
    loss = 0.0
    loss_epoch=[]
    iterations = int(training_samples/batch_size)
    for epoch in range(epochs):
        loss = 0.0
        for j in range (iterations):
            #print(j)
            data = dataloader.load_batch_train(batch_size)
            input = torch.tensor(data, dtype=torch.float32)
            optimizer.zero_grad()
            output = model.forward(input)
            loss_func = torch.mean((output - input) ** 2)
            loss += loss_func.item()
            loss_func.backward()
            optimizer.step()
            if j == (iterations-1) and (epoch%50==0 or epoch== epochs-1):
                x1 = output[0][0].view(1,64,64)
                x1 = x1.permute(1, 2, 0)
                x1 = x1.detach().numpy()
                plt.figure(epoch)
                plt.clf()
                plt.title('output')
                plt.imshow(x1, cmap="gray")
                #plt.pause(1)
                #plt.draw()
                plt.savefig("endofepoch" + str(epoch) +").png")
        loss_epoch.append(loss)
    return loss_epoch

training_loss = fit(model,data_loader)
print(training_loss)
#time.sleep(10)
'''



batch = data_loader.load_batch_train(50)
input = torch.tensor(batch, dtype=torch.float32)

x = model.compression(input)
x1 = x[0][0].view(1,8,8)
x1 = x1.permute(1, 2, 0)
x1 = x1.detach().numpy()
x1[x1 < 0] = 0
x1[x1 > 255] = 255
print(x1.shape)
plt.figure(1)
plt.clf()
plt.title('our method')
plt.imshow(x1, cmap="gray")
plt.show()
#print(x1)
#output = model.forward(input)
#'#''
'
for j in range(3):
    for i in range(20):
        #first of every 20 is the same video
        clip = batch[i][j]
        #clip = 255 - clip
        plt.figure(1)
        plt.clf()
        plt.title('our method')
        plt.imshow(clip, cmap='gray')
        plt.pause(1)
        plt.draw()  
'''
