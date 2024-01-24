from globals import *
import torch
from torch.autograd import Variable
from tqdm import tqdm


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(m.bias, 0.0)

class Generator(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(1024, 784),
            torch.nn.Tanh()
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(m.bias, 0.0)


def train_gan_model(generator, discriminator, train_set, optimizer_generator, optimizer_discriminator, loss):
    for epoch in range(num_epochs):
        for i, (images, real_labels) in enumerate(tqdm(train_set)):
            batch_size = images.size(0)

            # Adversarial ground truth
            fake_labels = Variable(torch.zeros(batch_size, 1))
            real_labels = Variable(torch.ones(batch_size, 1))

            # Configure input
            real_images = Variable(images.view(batch_size, -1))
# ----------------------------------------------------------------
# Training the Generator
# ----------------------------------------------------------------
            optimizer_generator.zero_grad()

            # Sample noise as generator input
            z = Variable(torch.zeros(batch_size, latent_dim))

            # Generate a batch of images
            fake_images = generator(z)
            
            # Loss measures generator's ability to fool the discriminator
            generator_loss = loss(discriminator(fake_images), real_labels)
            generator_loss.backward()
            optimizer_generator.step()
        
# ----------------------------------------------------------------
# Training the discriminator
# ----------------------------------------------------------------
            optimizer_discriminator.zero_grad()

            # Measures discriminator's ability to classify real from generated samples
            real_loss = loss(discriminator(real_images), real_labels)
            fake_loss = loss(discriminator(fake_images.detach()), fake_labels)
            discriminator_loss = (0.2*real_loss + 0.8*fake_loss) / 2
            discriminator_loss.backward()
            optimizer_discriminator.step()
            

        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"D_real_loss: {real_loss.item():.4f}, D_fake_loss: {fake_loss.item():.4f}, D_loss: {discriminator_loss.item():.4f} G_loss: {generator_loss.item():.4f}")