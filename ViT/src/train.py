import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_data, get_accuracy, plot_metrics
from model import VIT


load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')
RAW_DATA_DIR = os.path.join(DATA_DIR, os.getenv('RAW_DATA_DIR'))
TRAIN_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, os.getenv('TRAIN_RAW_DATA_DIR'))
TEST_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, os.getenv('TEST_RAW_DATA_DIR'))
MODELS_SAVE_DIR = os.getenv('MODELS_SAVE_DIR')
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE'))
IMAGE_CHANNELS = int(os.getenv('IMAGE_CHANNELS'))
VAL_PERCENT = float(os.getenv('VAL_PERCENT'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
PATCH_SIZE = int(os.getenv('PATCH_SIZE'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
NUM_HEADS = int(os.getenv('NUM_HEADS'))
NUM_ENCODERS = int(os.getenv('NUM_ENCODERS'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 


def train_epoch(epoch, model, data_loader, criterion, optimizer, device):
    '''
    Training epoch

    Args:
        epoch(int): Epoch number
        model(torch.nn.Module): Model to train
        data_loader(torch.utils.data.DataLoader): DataLoader object for training data
        criterion(torch.nn.Module): Loss function
        optimizer(torch.optim.Optimizer): Optimizer
        device(torch.device): Device to run the model

    Returns:
        running_loss(float): Running loss of the model
        running_acc(float): Running accuracy of the model
        losses(list): List of losses
        accs(list): List of accuracies
    '''
    model.train()
    running_loss = 0.0
    running_acc = 0
    
    loop = tqdm(data_loader)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        acc = get_accuracy(outputs, labels)
        running_acc += acc.item()
        loop.set_description(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        loop.set_postfix(loss=loss.item(), accuracy=acc.item())

    return running_loss / len(data_loader), running_acc / len(data_loader)


def val_epoch(epoch, model, data_loader, criterion, device):
    '''
    Validation epoch

    Args:
        epoch(int): Epoch number
        model(torch.nn.Module): Model to validate
        data_loader(torch.utils.data.DataLoader): DataLoader object for validation data
        criterion(torch.nn.Module): Loss function
        device(torch.device): Device to run the model

    Returns:
        running_loss(float): Running loss of the model
        running_acc(float): Running accuracy of the model
        losses(list): List of losses
        accs(list): List of accuracies
    '''
    model.eval()
    running_loss = 0.0
    running_acc = 0

    loop = tqdm(data_loader)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            acc = get_accuracy(outputs, labels)
            running_acc += acc.item()
            loop.set_description(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
            loop.set_postfix(loss=running_loss / len(data_loader), accuracy=acc.item())

    return running_loss / len(data_loader), running_acc / len(data_loader)


def training(plot=False):
    '''
    Train the models and save the loss and accuracy

    Args:
        plot(bool): Whether to plot the metrics or not

    Returns:
        loss_acc(dict): Dictionary containing the loss and accuracy of the models
    '''
    train_loader, val_loader, _, classes = load_data(TRAIN_RAW_DATA_DIR, TEST_RAW_DATA_DIR, IMAGE_SIZE, IMAGE_CHANNELS, BATCH_SIZE, VAL_PERCENT)

    model = VIT(IMAGE_SIZE, IMAGE_CHANNELS, PATCH_SIZE, embed_dim=768, 
            num_heads=NUM_HEADS, num_layers=NUM_ENCODERS, num_classes=len(classes), mlp_dropout=0.1, bias=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = val_epoch(epoch, model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    if plot:
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    torch.save(model.state_dict(), os.path.join(MODELS_SAVE_DIR, 'vit.pth'))