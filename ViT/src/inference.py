import os
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import get_accuracy, load_data, plot_conf_matrix
from sklearn.metrics import classification_report
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
NUM_HEADS = int(os.getenv('NUM_HEADS'))
NUM_ENCODERS = int(os.getenv('NUM_ENCODERS'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model(model, data_loader, criterion, device):
    '''
    Test the model

    Args:
        model(torch.nn.Module): Model to test
        data_loader(torch.utils.data.DataLoader): DataLoader object for testing data
        criterion(torch.nn.Module): Loss function
        device(torch.device): Device to run the model

    Returns:
        running_loss(float): Running loss of the model
        running_acc(float): Running accuracy of the model
        preds(list): Predictions from the model
        targets(list): True labels
    '''
    model.eval()
    running_loss = 0.0
    running_acc = 0
    preds = []
    targets = []
    loop = tqdm(data_loader)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            acc = get_accuracy(outputs, labels)
            running_acc += acc.item()
            loop.set_postfix(loss=running_loss / len(data_loader), accuracy=acc.item())
            preds += torch.argmax(outputs, 1).tolist()
            targets += labels.tolist()

    return running_loss / len(data_loader), running_acc / len(data_loader), preds, targets


def testing(plot_conf=False):
    '''
    Test the model

    Args:
        plot_conf(bool): Plot confusion matrix if True
    '''
    _, _, test_loader, classes = load_data(TRAIN_RAW_DATA_DIR, TEST_RAW_DATA_DIR, IMAGE_SIZE, IMAGE_CHANNELS, BATCH_SIZE, VAL_PERCENT)

    model = VIT(IMAGE_SIZE, IMAGE_CHANNELS, PATCH_SIZE, embed_dim=768, 
            num_heads=NUM_HEADS, num_layers=NUM_ENCODERS, num_classes=len(classes), mlp_dropout=0.1, bias=False).to(device)
    
    model.load_state_dict(torch.load(os.path.join(MODELS_SAVE_DIR, 'vit.pth')))
    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy, preds, targets = test_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    if plot_conf:
        plot_conf_matrix(preds, targets, classes)

    print(classification_report(targets, preds, target_names=classes))