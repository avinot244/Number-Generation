import torch
from torchvision.utils import save_image
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm
from generator import Generator, train_generator
from discriminator import Discriminator, train_discriminator
from globals import *
from data_loader import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
    generator = Generator(latent_dim)
    discriminator = Discriminator()

    optimizer_g = optim.Adam(generator.parameters(), lr=params.get("learning_rate_g"), betas=(params.get("optimizer_betas")))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=params.get("learning_rate_d"), betas=(params.get("optimizer_betas")))

    train_loader = load_data("./data")

    noise = create_noise(sample_size, latent_dim)
    generator.train()
    discriminator.train()

    for epoch in range(nb_epochs):
        loss_g = 0.0
        loss_d_real = 0.0
        loss_d_fake = 0.0

        # training
        for bi, data in enumerate(tqdm(train_loader)):
            image, _ = data
            b_size = len(image)

            for step in range(k):
                data_fake = generator(create_noise(b_size, latent_dim)).detach()
                data_real = image
                loss_d_fake_real = train_discriminator(optimizer_d, data_real, data_fake, discriminator)
                loss_d_real += loss_d_fake_real[0]
                loss_d_fake += loss_d_fake_real[1]
            data_fake = generator(create_noise(b_size, latent_dim))
            loss_g += train_generator(optimizer_g, data_fake, discriminator)
        
        # inference and observations
        generated_img = generator(noise)
        generated_img = generated_img.view(generated_img.size(0), 1, 28, 28)
        save_image(generated_img, f"./generated_img/generated_img{epoch}.png", nrow=8, normalize=True)
        epoch_loss_g = loss_g / bi
        epoch_loss_d_real = loss_d_real/bi
        epoch_loss_d_fake = loss_d_fake/bi
        print(f"Epoch {epoch} of {nb_epochs}")
        print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss fake: {epoch_loss_d_fake:.8f}, Discriminator loss real: {epoch_loss_d_real:.8f}")

torch.save(generator.state_dict(), f"./model/saved_models/{nb_epochs}/gan_model_{nb_epochs}")
plt.plot(range(nb_epochs), epoch_loss_g, label="Generator")
plt.plot(range(nb_epochs), epoch_loss_d_fake, label="Discriminator fake")
plt.plot(range(nb_epochs), epoch_loss_d_real, label="Discriminator real")