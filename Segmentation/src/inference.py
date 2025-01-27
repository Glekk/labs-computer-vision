import os
from dotenv import load_dotenv
from tqdm import tqdm
import torch
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model(model, metric, loader):
    '''
    Testing the model

    Args:
        model(torch.nn.Module): Model to test
        metric(torch.nn.Module): Metric function
        loader(torch.utils.data.DataLoader): DataLoader object for testing set

    Returns:
        running_iou(float): Average IoU
        img_list(list): List of images
        preds_list(list): List of predictions
        labels_list(list): List of labels (masks)
    '''
    model.eval()
    running_iou = 0.0
    img_list = []
    preds_list = []
    labels_list = []
    loop = tqdm(loader)
    with torch.no_grad():
        for img, label in loop:
            img, label = img.to(device), label.to(device)
            preds = model(img)
            running_iou += metric(preds, label)
            img_list.append(img)
            preds_list.append(preds)
            labels_list.append(label)

    return running_iou / len(loader), torch.cat(img_list), torch.cat(preds_list), torch.cat(labels_list)


def testing():
    '''
    Testing the model

    Returns:
        imgs(torch.Tensor): Images
        preds(torch.Tensor): Predictions
        labels(torch.Tensor): Labels (masks)
    '''
    _, _, test_loader = load_data(IMG_DIR, MASK_DIR, BATCH_SIZE, IMAGE_SIZE, TRAIN_SIZE, VAL_TEST_SPLIT, seed=SEED)
    model = DeepLabV3Plus(1).to(device)  
    iou_metric = IoU()

    model.load_state_dict(torch.load(os.path.join(MODELS_SAVE_DIR, 'deeplabv3plus.pth')))
    iou, imgs, preds, labels = test_model(model, iou_metric, test_loader)
    print(f'Test IoU: {iou:.4f}')

    return imgs, preds, labels