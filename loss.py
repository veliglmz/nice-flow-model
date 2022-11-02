import torch
import numpy as np


def gaussian_distribution_loss(hd, diagonal):
    loss = torch.sum(diagonal) - (0.5*torch.sum(torch.pow(hd,2),dim=1) + hd.size(1)*0.5*torch.log(torch.tensor(2*np.pi)))
    loss = torch.mean(loss)
    return -loss