import os
from dotenv import load_dotenv
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_data, DiceLoss, IoU
from model import DeepLabV3Plus

load_dotenv()
SEED = int(os.getenv('SEED'))
DATA_DIR = os.getenv('DATA_DIR')
RAW_DIR = os.path.join(DATA_DIR, os.getenv('RAW_DIR'))
MODELS_SAVE_DIR = os.getenv('MODELS_SAVE_DIR')
IMG_DIR = os.path.join(RAW_DIR, os.getenv('IMG_DIR'))
MASK_DIR = os.path.join(RAW_DIR, os.getenv('MASK_DIR'))
TRAIN_SIZE = int(os.getenv('TRAIN_SIZE'))
VAL_TEST_SPLIT = float(os.getenv('VAL_TEST_SPLIT'))
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 

def train_epoch(model, loader, criterion, metric, optimizer, epoch, num_epochs):
    '''
    Training epoch

    Args:
        model(torch.nn.Module): Model to train
        loader(torch.utils.data.DataLoader): DataLoader object for training set
        criterion(torch.nn.Module): Loss function
        metric(torch.nn.Module): Metric function
        optimizer(torch.optim.Optimizer): Optimizer
        epoch(int): Current epoch
        num_epochs(int): Number of epochs

    Returns:
        running_loss(float): Average loss
        running_iou(float): Average IoU
    '''
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    loop = tqdm(loader)
    for img, label in loop:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        preds = model(img)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iou = metric(preds, label).item()
        running_iou += iou
        loop.set_description(f'Epoch {epoch + 1}/{num_epochs}')
        loop.set_postfix(loss=loss.item(), iou=iou)

    return running_loss / len(loader), running_iou / len(loader)


def val_epoch(model, loader, criterion, metric, epoch, num_epochs):
    '''
    Validation epoch

    Args:
        model(torch.nn.Module): Model to train
        loader(torch.utils.data.DataLoader): DataLoader object for validation set
        criterion(torch.nn.Module): Loss function
        metric(torch.nn.Module): Metric function
        epoch(int): Current epoch
        num_epochs(int): Number of epochs

    Returns:
        running_loss(float): Average loss
        running_iou(float): Average IoU
    '''
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    loop = tqdm(loader)
    with torch.no_grad():
        for img, label in loop:
            img, label = img.to(device), label.to(device)
            preds = model(img)
            loss = criterion(preds, label)
            running_loss += loss.item()
            iou = metric(preds, label).item()
            running_iou += iou
            loop.set_description(f'Epoch {epoch + 1}/{num_epochs}')
            loop.set_postfix(loss=loss.item(), accuracy=iou)

    return running_loss / len(loader), running_iou / len(loader)


def train_model(model, train_loader, val_loader, criterion, metric, optimizer, num_epochs):
    '''
    Main training loop

    Args:
        model(torch.nn.Module): Model to train
        train_loader(torch.utils.data.DataLoader): DataLoader object for training set
        val_loader(torch.utils.data.DataLoader): DataLoader object for validation set
        criterion(torch.nn.Module): Loss function
        metric(torch.nn.Module): Metric function
        optimizer(torch.optim.Optimizer): Optimizer
        num_epochs(int): Number of epochs

    Returns:
        train_losses(list): List of training losses
        train_ious(list): List of training IoUs
        val_losses(list): List of validation losses
        val_ious(list): List of validation IoUs
    '''
    train_losses = []
    train_ious = []
    val_losses = []
    val_ious = []
    for epoch in range(num_epochs):
        train_loss, train_iou = train_epoch(model, train_loader, criterion, metric, optimizer, epoch, num_epochs)
        val_loss, val_iou = val_epoch(model, val_loader, criterion, metric, epoch, num_epochs)
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        print(f'Training Loss: {train_loss:.4f}, Training IoU: {train_iou:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}')
    return train_losses, train_ious, val_losses, val_ious


def training():
    '''
    Train the model and save the metrics and the model
    '''
    model = DeepLabV3Plus(1).to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    iou_metric = IoU()

    train_loader, val_loader, _ = load_data(IMG_DIR, MASK_DIR, BATCH_SIZE, IMAGE_SIZE, TRAIN_SIZE, VAL_TEST_SPLIT, seed=SEED)

    train_losses, train_ious, val_losses, val_ious = train_model(model, train_loader, val_loader, criterion, iou_metric, optimizer, NUM_EPOCHS)

    with open(os.path.join(MODELS_SAVE_DIR, 'metrics.pkl'), 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'train_ious': train_ious,
            'val_losses': val_losses,
            'val_ious': val_ious
        }, f)

    torch.save(model.state_dict(), os.path.join(MODELS_SAVE_DIR, 'deeplabv3plus.pth'))