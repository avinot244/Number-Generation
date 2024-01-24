import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from globals import *

def load_data(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    mnist_dataset = datasets.MNIST(root=path, train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    return dataloader