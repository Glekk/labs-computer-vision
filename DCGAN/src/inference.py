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


def test_model():
    '''
    Test the generator model

    Returns:
        fake (tensor): Generated images
    '''
    gen, _ = get_model(IMAGE_CHANNELS, Z_DIM, GEN_FEATURES, DISCR_FEATURES)

    gen.load_state_dict(torch.load(os.path.join(MODELS_SAVE_DIR, "gen.pth")))
    gen.eval()
    gen.to(device)

    noise = torch.randn(64, Z_DIM, 1, 1, device=device)
    fake = gen(noise)
    show_tensor_images(fake, title="Generated Images", figsize=(14, 14))

    return fake
