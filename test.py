import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

DIR = './dataset/'

image_size = 64
batch_size = 64

train_ds = ImageFolder(DIR, transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()]))

train_dl = DataLoader(train_ds, batch_size)
def show_images(images):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images.detach(), nrow=8).permute(1, 2, 0))

def show_batch(dl):
    for images, _ in dl:
        show_images(images)
        break

show_batch(train_dl)

