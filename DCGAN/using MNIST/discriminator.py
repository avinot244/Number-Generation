import torch.nn as nn
from globals import *


class Discriminator(nn.Module):
   def __init__(self):
       super(Discriminator, self).__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
           nn.LeakyReLU(0.2),

           nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
           nn.LeakyReLU(0.2)
       )
       self.classifier = nn.Sequential(
           nn.Linear(in_features=3136, out_features=1),
           nn.Sigmoid()
       )

   def forward(self, x):
       x = self.conv(x)
       x = x.view(x.size(0), -1)
       x = self.classifier(x)
       return x
   
def train_discriminator(optimizer, data_real, data_fake, discriminator):
    b_size = data_real.size(0)
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)
    optimizer.zero_grad()
    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)
    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()
    return loss_real, loss_fake