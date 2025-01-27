import os
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_data, get_model, show_tensor_images


load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')
RAW_DATA_DIR = os.path.join(DATA_DIR, os.getenv('RAW_DATA_DIR'))
MODELS_SAVE_DIR = os.getenv('MODELS_SAVE_DIR')
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE'))
IMAGE_CHANNELS = int(os.getenv('IMAGE_CHANNELS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
Z_DIM = int(os.getenv('Z_DIM'))
GEN_FEATURES = int(os.getenv('GEN_FEATURES'))
DISCR_FEATURES = int(os.getenv('DISCR_FEATURES'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training(show_images_on_epoch=False):
    '''
    Train the GAN model

    Args:
        show_images_on_epoch (bool): If True, show generated images on each epoch. Default is False.
    '''
    dataloader = load_data(RAW_DATA_DIR, IMAGE_SIZE, IMAGE_CHANNELS, BATCH_SIZE)
    gen, disc = get_model(IMAGE_CHANNELS, Z_DIM, GEN_FEATURES, DISCR_FEATURES)

    gen_optim = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    disc_optim = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=device)
    gen_losses = []
    disc_losses = []
    mean_gen_losses = []
    mean_disc_losses = []

    for epoch in range(NUM_EPOCHS):

        loop = tqdm(dataloader, leave=True)
        gen.train()
        disc.train()

        for idx, (real, _) in enumerate(loop):
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
            fake = gen(noise)

            disc.zero_grad()
            disc_real = disc(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            loss_disc.backward()
            disc_optim.step()

            gen_optim.zero_grad()
            output = disc(fake).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            
            loss_gen.backward()
            gen_optim.step()

            gen_losses.append(loss_gen.item())
            disc_losses.append(loss_disc.item())

            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss_disc=loss_disc.item(), loss_gen=loss_gen.item())

        mean_gen_losses.append(np.mean(gen_losses[-len(dataloader):]))
        mean_disc_losses.append(np.mean(disc_losses[-len(dataloader):]))
        print(f"Mean Generator Loss: {mean_gen_losses[-1]}, Mean Discriminator Loss: {mean_disc_losses[-1]}")

        if show_images_on_epoch:
            gen.eval()
            with torch.no_grad():
                fake = gen(fixed_noise)
                show_tensor_images(fake)

    torch.save(gen.state_dict(), os.path.join(MODELS_SAVE_DIR, "gen.pth"))
    torch.save(disc.state_dict(), os.path.join(MODELS_SAVE_DIR, "discr.pth"))