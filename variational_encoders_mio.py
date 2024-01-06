import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torchvision.datasets as datasets
import numpy as np

from models import VAE
from config import *


#Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



transformed = transforms.Compose([transforms.ToTensor()])

#creamos el set de train y de test
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transformed)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transformed)

train_data = dataloader.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) 
test_data = dataloader.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    

model = VAE(conditioned=True).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
images = []

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, label) in enumerate(train_data):
        data = data.to(device)
        label = label.to(device)
        mu, sigma = model.encode(data)
        z = model.reparametrizacion(mu, sigma)
        recon = model.decode(z, label)
        
        #definimos la loss de kulback leiber
        kld_loss = torch.mean(-0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), dim = 1))

        
        loss = 500*criterion(recon, data) + kld_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Epoch {}, Batch idx {}, loss {}'.format(epoch, batch_idx, loss.item()))

# test loop
ruta_guardado = 'VAE.pth'

# Guarda el modelo
torch.save(model.state_dict(), ruta_guardado)


model.eval()
for example in range(10):
    prova = model.sample(18, device, torch.ones(18)*example)
    torchvision.utils.save_image(prova, './images_vae/recon_images_test_prova_{}.png'.format(example), nrow=4) 
    

        