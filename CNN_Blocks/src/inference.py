import os
from dotenv import load_dotenv
from tqdm import tqdm
import torch
from utils import get_acc, load_data, get_models
from sklearn.metrics import classification_report

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model(model, loader):
    '''
    Test the model on the given loader

    Args:
        model(torch.nn.Module): Model to test
        loader(torch.utils.data.DataLoader): Loader for the test data

    Returns:
        accuracy(float): Accuracy of the model
        preds(torch.Tensor): Predictions of the model
        labels(torch.Tensor): True labels
    '''
    model.eval()
    running_accuracy = 0.0
    preds_list = []
    labels_list = []
    loop = tqdm(loader)
    with torch.no_grad():
        for img, label in loop:
            img, label = img.to(device), label.to(device)
            preds = model(img)
            running_accuracy += get_acc(preds, label)
            preds_list.append(preds)
            labels_list.append(label)

    return running_accuracy / len(loader), torch.cat(preds_list), torch.cat(labels_list)


def testing_all():
    '''
    Test all the models

    Returns:
        test_results(dict): Dictionary of test results
    '''
    _, _, test_loader, classes = load_data(TRAIN_DIR, TEST_DIR, IMAGE_SIZE, IMAGE_CHANNELS, BATCH_SIZE, VAL_PERCENT)
    standard_cnn, se_cnn, depthwise_cnn, resbottleneck_cnn = get_models(IMAGE_CHANNELS, classes, device)

    test_results = {}
    for model, name in zip([standard_cnn, se_cnn, depthwise_cnn, resbottleneck_cnn], ['standard_cnn', 'se_cnn', 'depthwise_cnn', 'resbottleneck_cnn']):
        model.load_state_dict(torch.load(f'{MODELS_SAVE_DIR}/{name}.pth'))
        accuracy, preds, labels = test_model(model, test_loader)
        test_results[name] = (accuracy, preds, labels)
        print(f'{name} accuracy: {accuracy:.4f}')

    for name in test_results:
        accuracy, preds, labels = test_results[name]
        print(f'{name} accuracy: {accuracy:.4f}')
        print(classification_report(labels.cpu(), preds.argmax(dim=1).cpu(), target_names=classes, zero_division=0))

    return test_results
