import torch
import random
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from globals import manualSeed, ngpu
from dataloader import getData


if __name__ == "__main__":
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    dataloader = getData()
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    print("Hello world!")