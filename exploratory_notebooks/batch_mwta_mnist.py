# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# %%
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)


batch_size = 4096

train_dataloader = DataLoader(training_data, batch_size=batch_size)


# %%
next(iter(train_dataloader))[0].shape
# %%
