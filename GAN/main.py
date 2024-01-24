import torch
import random
from load_data import load_data
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
from globals import *
from gan_model import *
import matplotlib.pyplot as plt
import os



if __name__ == "__main__" :
    epochs = num_epochs
    
    generator = Generator(latent_dim)
    discriminator = Discriminator()
    train_set = load_data('../data')


    loss = torch.nn.BCELoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr_d)
    
    train_gan_model(generator, discriminator, train_set, optimizer_generator, optimizer_discriminator, loss)

    if not(os.path.exists(f"{PROJECT_PATH}/saved_models/")):
        os.mkdir(f"{PROJECT_PATH}/saved_models/")
    assert os.path.exists(f"{PROJECT_PATH}/saved_models/")

    if not(os.path.exists(f"{PROJECT_PATH}/saved_models/{epochs}/")):
        os.mkdir(f"{PROJECT_PATH}/saved_models/{epochs}/")
    assert os.path.exists(f"{PROJECT_PATH}/saved_models/{epochs}/")
    # torch.save(generator.state_dict(), f"{PROJECT_PATH}/saved_models/{epochs}/gan_model_{epochs}")
    
    examples = enumerate(train_set)
    _, (example_data, example_targets) = next(examples)

    with torch.no_grad():
        output = discriminator(Variable(example_data.view(batch_size, -1)))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    noise = Variable(torch.randn(100, latent_dim))
    generated_images = generator(noise)
    generated_images = generated_images.view(generated_images.size(0), 1, 28, 28)
    save_image(generated_images.data, f'{PROJECT_PATH}/saved_models/{epochs}/generated_images.png', nrow=10, normalize=True)