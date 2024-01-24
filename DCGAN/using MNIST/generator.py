import torch.nn as nn
from globals import *

class Generator(nn.Module):
   def __init__(self, latent_space):
       super(Generator, self).__init__()
       self.latent_space = latent_space
       self.fcn = nn.Sequential(
           nn.Linear(in_features=self.latent_space, out_features=128*7*7),
           nn.LeakyReLU(0.2),
       )

       self.deconv = nn.Sequential(
           nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
           nn.LeakyReLU(0.2),

           nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
           nn.LeakyReLU(0.2),

           nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(3, 3), padding=(1, 1)),
           nn.Tanh()
       )

   def forward(self, x):
       x = self.fcn(x)
       x = x.view(-1, 128, 7, 7)
       x = self.deconv(x)
       return x

def train_generator(optimizer, data_fake, discriminator):
    b_size = data_fake.size(0)
    real_label = label_real(b_size)
    optimizer.zero_grad()
    output = discriminator(data_fake)
    loss = criterion(output, real_label)
    loss.backward()
    optimizer.step()
    return loss