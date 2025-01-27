import os
from dotenv import load_dotenv
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_data, get_acc, get_models


load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')
RAW_DIR = os.path.join(DATA_DIR, os.getenv('RAW_DIR'))
MODELS_SAVE_DIR = os.getenv('MODELS_SAVE_DIR')
TRAIN_DIR = os.path.join(RAW_DIR, os.getenv('TRAIN_DIR'))
TEST_DIR = os.path.join(RAW_DIR, os.getenv('TEST_DIR'))
VAL_PERCENT = float(os.getenv('VAL_PERCENT'))
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE'))
IMAGE_CHANNELS = int(os.getenv('IMAGE_CHANNELS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 

def train_epoch(model, loader, criterion, optimizer, epoch, num_epochs):
    '''
    Train one epoch of the model

    Args:
        model(torch.nn.Module): Model to train
        loader(torch.utils.data.DataLoader): Data loader for the training data
        criterion(torch.nn.Module): Loss function
        optimizer(torch.optim.Optimizer): Optimizer
        epoch(int): Current epoch
        num_epochs(int): Total number of epochs

    Returns:
        running_loss(float): Average loss of the model
        running_accuracy(float): Average accuracy of the model
    '''
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    loop = tqdm(loader)
    for img, label in loop:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        preds = model(img)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        acc = get_acc(preds, label).item()
        running_accuracy += acc
        loop.set_description(f'Epoch {epoch + 1}/{num_epochs}')
        loop.set_postfix(loss=loss.item(), accuracy=acc)

    return running_loss / len(loader), running_accuracy / len(loader)


def val_epoch(model, loader, criterion, epoch, num_epochs):
    '''
    Validate one epoch of the model

    Args:
        model(torch.nn.Module): Model to validate
        loader(torch.utils.data.DataLoader): Data loader for the validation data
        criterion(torch.nn.Module): Loss function
        epoch(int): Current epoch
        num_epochs(int): Total number of epochs

    Returns:
        running_loss(float): Average loss of the model
        running_accuracy(float): Average accuracy of the model
    '''
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    loop = tqdm(loader)
    with torch.no_grad():
        for img, label in loop:
            img, label = img.to(device), label.to(device)
            preds = model(img)
            loss = criterion(preds, label)
            running_loss += loss.item()
            acc = get_acc(preds, label).item()
            running_accuracy += acc
            loop.set_description(f'Epoch {epoch + 1}/{num_epochs}')
            loop.set_postfix(loss=loss.item(), accuracy=acc)

    return running_loss / len(loader), running_accuracy / len(loader)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    '''
    Main training loop

    Args:
        model(torch.nn.Module): Model to train
        train_loader(torch.utils.data.DataLoader): Data loader for the training data
        val_loader(torch.utils.data.DataLoader): Data loader for the validation data
        criterion(torch.nn.Module): Loss function
        optimizer(torch.optim.Optimizer): Optimizer
        num_epochs(int): Total number of epochs

    Returns:
        train_losses(list): List of training losses
        train_accuracies(list): List of training accuracies
        val_losses(list): List of validation losses
        val_accuracies(list): List of validation accuracies
    '''
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, epoch, num_epochs)
        val_loss, val_accuracy = val_epoch(model, val_loader, criterion, epoch, num_epochs)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    return train_losses, train_accuracies, val_losses, val_accuracies


def training():
    '''
    Train the models and save the loss and accuracy

    Returns:
        loss_acc(dict): Dictionary containing the loss and accuracy of the models
    '''
    train_loader, val_loader, test_loader, classes = load_data(TRAIN_DIR, TEST_DIR, IMAGE_SIZE, IMAGE_CHANNELS, BATCH_SIZE, VAL_PERCENT)
    standard_cnn, se_cnn, depthwise_cnn, resbottleneck_cnn = get_models(IMAGE_CHANNELS, classes, device)


    loss_acc = {}

    for model, name in zip([standard_cnn, se_cnn, depthwise_cnn, resbottleneck_cnn], ['standard_cnn', 'se_cnn', 'depthwise_cnn', 'resbottleneck_cnn']):
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
        loss_acc[name] = (train_losses, train_accuracies, val_losses, val_accuracies)
        with open(f'{MODELS_SAVE_DIR}/{name}_loss_acc.pkl', 'wb') as f:
            pickle.dump(loss_acc, f)
        torch.save(model.state_dict(), f'{MODELS_SAVE_DIR}/{name}.pth')
        
    return loss_acc