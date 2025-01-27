import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


def mask_preprocess(mask):
    '''Preprocess mask to have values 0 or 1'''
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    return mask


class CustomDataset(Dataset):
    '''Dataset class for images and masks'''
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        mask_path = self.data.iloc[idx]['mask_path']

        image = plt.imread(img_path)
        mask = plt.imread(mask_path)
        mask = mask_preprocess(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        mask = mask.unsqueeze(0)
        
        return image, mask
    

class DiceLoss(nn.Module):
    '''Dice loss function'''
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        return 1 - ((2. * intersection + 1e-7) / (y_pred.sum() + y_true.sum() + 1e-7))
    

class IoU(nn.Module):
    '''Intersection over Union metric'''
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum() - intersection
        return (intersection + 1e-7) / (union + 1e-7)
    

def get_paths(img_dir, mask_dir):
    images_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
    masks_paths = [os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir)]

    images_paths.sort()
    masks_paths.sort()

    print(f'Images: {len(images_paths)}')
    print(f'Masks: {len(masks_paths)}')

    data_df = pd.DataFrame({
        'image_path': images_paths,
        'mask_path': masks_paths
    })
    
    return data_df


def clean_data(data_df):
    '''
    Remove images and masks with different shapes

    Args:
        data_df(pd.DataFrame): DataFrame with images and masks paths
    '''
    images_paths = data_df['image_path'].tolist()
    masks_paths = data_df['mask_path'].tolist()

    for im, msk in zip(images_paths, masks_paths):
        img = plt.imread(im)
        mask = plt.imread(msk)
        if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
            print(img.shape, mask.shape)
            print(im, msk)
            print(f'Index {images_paths.index(im)}')
            os.remove(im)
            os.remove(msk)


def load_data(img_dir, mask_dir, batch_size, image_size, train_size, val_test_split, clean_data=False, seed=42):
    '''
    Load data from directories and split it into train, validation and test sets

    Args:
        img_dir(str): Path to images directory
        mask_dir(str): Path to masks directory
        batch_size(int): Batch size
        image_size(int): Image size
        train_size(float): Train size
        val_test_split(float): Validation and test split size
        clean_data(bool): If True, remove images and masks with different shapes
        seed(int): Random seed

    Returns:
        train_loader(torch.utils.data.DataLoader): DataLoader object for training set
        val_loader(torch.utils.data.DataLoader): DataLoader object for validation set
        test_loader(torch.utils.data.DataLoader): DataLoader object for test set
    '''
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2()
    ])

    data_df = get_paths(img_dir, mask_dir)

    if clean_data:
        clean_data(data_df)

    train_df, other_df = train_test_split(data_df, train_size=train_size, random_state=seed)
    val_df, test_df = train_test_split(other_df, test_size=val_test_split, random_state=seed)

    train = CustomDataset(train_df, transform=train_transform)
    val = CustomDataset(val_df, transform=val_transform)
    test = CustomDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def show_images(loader, cols=2, rows=2):
    '''
    Plot images and masks from the loader

    Args:
        loader(torch.utils.data.DataLoader): DataLoader object
        cols(int): Number of columns
        rows(int): Number of rows
    '''
    images_sample = next(iter(loader))

    fig = plt.figure(figsize=(8, 8))

    for i in range(cols*rows):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(np.clip(images_sample[0][i].permute(1, 2, 0).numpy()*[0.229, 0.224, 0.225]+[0.485, 0.456, 0.406], 0, 1))
        plt.imshow(images_sample[1][i][0].numpy(), alpha=0.2)
        plt.axis('off')
    plt.show()


def plot_metrics(train_losses, train_ious, val_losses, val_ious, figsize=(8, 4), cols=2, rows=1):
    '''
    Plot training and validation metrics

    Args:
        train_losses(list): List of training losses
        train_ious(list): List of training IoU scores
        val_losses(list): List of validation losses
        val_ious(list): List of validation IoU scores
        figsize(tuple): Figure size
        cols(int): Number of columns
        rows(int): Number of rows
    '''
    fig = plt.figure(figsize=figsize)

    for i, (train, val) in enumerate([(train_losses, val_losses), (train_ious, val_ious)]):
        fig.add_subplot(rows, cols, i+1)
        plt.plot(train, label='Training')
        plt.plot(val, label='Validation')
        plt.legend()

        plt.title(f'Loss' if i == 0 else f'Accuracy')
    plt.show()


def plot_predictions(img, pred, true, figsize=(15, 10)):
    '''
    Plot image, prediction and true mask

    Args:
        img(torch.Tensor): Image tensor
        pred(torch.Tensor): Prediction tensor
        true(torch.Tensor): True mask tensor
    '''
    fig = plt.figure(figsize=figsize)

    fig.add_subplot(1, 3, 1)
    plt.imshow(np.clip(img.permute(1, 2, 0).detach().cpu().numpy()*[0.229, 0.224, 0.225]+[0.485, 0.456, 0.406], 0, 1))
    plt.axis('off')
    plt.title('Image')
    fig.add_subplot(1, 3, 2)
    plt.imshow(mask_preprocess(torch.sigmoid(pred[0]).detach().cpu().numpy()), cmap='gray')
    plt.axis('off')
    plt.title('Prediction')
    fig.add_subplot(1, 3, 3)
    plt.imshow(true[0].detach().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('True Mask')
    plt.show()