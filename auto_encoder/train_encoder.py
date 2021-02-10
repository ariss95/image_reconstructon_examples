import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch_encoder
import matplotlib.pyplot as plt

batch_size = 64
epochs = 40
learning_rate = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data = datasets.MNIST(
    root='input/data',
    train=True,
    download=True,
    transform=transform
)
val_data = datasets.MNIST(
    root='input/data',
    train=False,
    download=True,
    transform=transform
)
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False
)

model = torch_encoder.Encoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss(reduction="sum")

def fit(model, dataloader):
    model.train()
    loss = 0.0
    for i, data in enumerate(dataloader, 0):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstructed = model(data)
        l = criterion(reconstructed, data)
        loss += l.item()
        l.backward()
        optimizer.step()
        if i == int(len(val_data)/dataloader.batch_size) - 1:
            save_image_to_disk(data, reconstructed,"training", epoch)

    train_loss = loss/len(dataloader.dataset)#TODO  

    return train_loss

def psnr(ref, reconstructed):
        reconstructed[reconstructed < 0] = 0
        reconstructed[reconstructed > 255] = 255
        reconstructed = reconstructed.int().float()
        ref = ref.int().float()
        ref = 255 - ref
        #print(ref)
        #print(reconstructed)
        mse = torch.mean((ref - reconstructed) ** 2)
        #fix
        if mse == 0:
            return 100
        return 10 * torch.log10(255 / torch.sqrt(mse))

def validate(model, dataloader):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstructed = model(data)
            l = criterion(reconstructed, data)
            #psnr_ = psnr(data, reconstructed)
            #print(psnr_.item())
            loss += l.item()
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                save_image_to_disk(data, reconstructed,"validation", epoch)
    val_loss = loss/len(dataloader.dataset)
    return val_loss

def save_image_to_disk(data, reconstructed, caller, i):
    print("saving")
    num_rows = 8
    filename = "../outputs/"+ caller + str(i) +".png"
    both = torch.cat((data.view(batch_size, 1, 28, 28)[:8], 
                        reconstructed.view(batch_size, 1, 28, 28)[:8]))
    save_image(both.cpu(), filename, nrow=num_rows)


'''
images, labels = next(iter(train_loader))
frame1 = images[0]
frame2 = images[1]
plt.figure(1)
plt.title(0)
plt.imshow(frame1.permute(1, 2, 0), cmap="gray")
plt.figure(2)
plt.title(1)
plt.imshow(frame2.permute(1, 2, 0), cmap="gray")

'''
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, train_loader)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    v = validate(model, val_loader)
    print(f"validation Loss: {v:.4f}")
