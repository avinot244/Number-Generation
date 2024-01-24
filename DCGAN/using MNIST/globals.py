import torch

batch_size = 1024
nb_epochs = 10
sample_size = 64
latent_dim = 128
lr_g = 0.0002
lr_d = 0.0002
k = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()
params = {"learning_rate_g": 0.0002,
          "learning_rate_d": 0.0002,
          "optimizer": "Adam",
          "optimizer_betas": (0.5, 0.999),
          "latent_dim": latent_dim}

def label_real(size):
    labels = torch.ones(size, 1)
    return labels

def label_fake(size):
    labels = torch.zeros(size, 1)
    return labels

def create_noise(sample_size, latent_dim):
    noise = torch.randn(sample_size, latent_dim)
    return noise

