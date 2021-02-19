import data_loader as dl
import model_RNN
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import plot_utils
path = '/home/aris/Desktop/anomaly_detection/movingMnist/mnist_test_seq_16.npy'
data_loader = dl.Moving_MNIST_Loader(path=path, time_steps=20, load_only=-1,
                                      flatten=True, scale=False)
device = torch.device('cpu')
model = model_RNN.first_RNN(256).to(device)
learning_rate = 0.001
epochs = 60
batch_size = 64
training_samples = 8000
test_samples = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def fit(model, dataloader):
	model.train()
	loss = 0.0
	loss_epoch=[]
	iterations = int(training_samples/batch_size)
	for epoch in range(epochs):
		loss = 0.0
		print("epoch: " + str(epoch))
		for j in range (iterations):
			#print(j)
			data = dataloader.load_batch_train(batch_size)
			input_ = torch.tensor(data, dtype=torch.float32, device=device)
			optimizer.zero_grad()
			output = model.forward(input_)
			loss_func = torch.mean((output - input_) ** 2)
			loss += loss_func.item()
			loss_func.backward()
			optimizer.step()
			if j == (iterations-1) and (epoch%50==0 or epoch== epochs-1):
				filename = "endofepoch" + str(epoch) +".png"
				plot_utils.save_frame(output, filename)
		print("epoch loss: " + str(loss))


def test(model, dataloader):
	
	model.eval()
	with torch.no_grad():
		for i in range (int(test_samples/batch_size)): 
			loss = 0.0
			test_data = dataloader.load_batch_test(batch_size)
			input_ = torch.tensor(test_data, dtype=torch.float32, device=device)
			output = model.forward(input_)
			loss = torch.mean((output - input_) ** 2).item()
			#TODO evaluation, maybe psnr
			plot_utils.save_sequences(input_, 0, output, "test_sequence0" + str(i) +".png")
			#plot_utils.save_sequences(input_, 1, output, "test_sequence1" + str(i) +".png")
			plot_utils.save_sequences(input_, batch_size -1 , output, "test_sequence64" + str(i) +".png")
			print ("test loss: " + str(loss))

fit(model,data_loader)

test(model, data_loader)


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
