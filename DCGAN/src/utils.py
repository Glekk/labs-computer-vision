import numpy as np
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model import Generator, Discriminator


def load_data(data_dir, image_size, image_channels, batch_size, num_workers=8):
    '''
    Load data from the given directories

    Args:
        data_dir(str): Directory containing the data
        image_size(int): Size of the image
        image_channels(int): Number of channels in the image
        batch_size(int): Batch size
        num_workers(int): Number of workers for the dataloader

    Returns:
        dataloader(torch.utils.data.DataLoader): Dataloader object
    '''
    transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5] * image_channels, [0.5] * image_channels)])

    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader

def show_images(dataloader, device, imgs_num=4, normalize=True, figsize=(8, 8)):
    '''
    Display images from the dataloader

    Args:
        dataloader(torch.utils.data.DataLoader): Dataloader object
        device(torch.device): Device for the images
        imgs_num(int): Number of images to display (should be less than batch size)
        normalize(bool): If True, shift the image to the range (0, 1), by the min and max values 
        figsize(tuple): Figure size
    '''
    images = next(iter(dataloader))
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(images[0].to(device)[:imgs_num], padding=2, normalize=normalize).cpu(), (1, 2, 0)))
    plt.show()

def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def get_model(image_channels, z_dim, gen_features, disc_features, device):
    '''
    Get the model

    Returns:
        model(nn.Module): Model object
    '''
    gen = Generator(z_dim, image_channels, gen_features).to(device)
    disc = Discriminator(image_channels, disc_features).to(device)
    init_weights(gen)
    init_weights(disc)

    return gen, disc

def show_tensor_images(images, figsize=(8, 8), title=None):
    '''
    Display images from the tensor

    Args:
        images(torch.Tensor): Tensor containing the images
        figsize(tuple): Figure size
        title(str): Title of the plot
    '''
    img_cpu = images.detach().cpu()
    img_grid = make_grid(img_cpu[:64], padding=2, normalize=True)
    plt.figure(figsize=figsize)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_losses(gen_losses, disc_losses):
    '''
    Plot the losses

    Args:
        gen_losses(list): List of generator losses
        disc_losses(list): List of discriminator losses
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.legend()
    plt.title("Losses")
    plt.show()