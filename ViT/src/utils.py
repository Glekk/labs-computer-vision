import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_data(train_dir, test_dir, image_size, image_channels, batch_size, val_percent=0.15, num_workers=8):
    '''
    Load data from the given directories

    Args:
        train_dir(str): Path to the training data
        test_dir(str): Path to the testing data
        image_size(int): Size of the image
        image_channels(int): Number of channels in the image
        batch_size(int): Batch size
        val_percent(float): Percentage of the data to be used for validation
        num_workers(int): Number of workers to load the data

    Returns:
        train_loader(torch.utils.data.DataLoader): DataLoader object for training data
        val_loader(torch.utils.data.DataLoader): DataLoader object for validation data
        test_loader(torch.utils.data.DataLoader): DataLoader object for testing data
        train.classes(list): List of class names
    '''
    train_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5] * image_channels, [0.5] * image_channels)])


    test_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5] * image_channels, [0.5] * image_channels)])
    
    train = ImageFolder(train_dir, transform=train_transform)
    test = ImageFolder(test_dir, transform=test_transform)

    val_size = int(val_percent * len(train))
    train_size = len(train) - val_size
    train, val = random_split(train, [train_size, val_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train.classes


def get_accuracy(preds, y):
    '''
    Get accuracy of the model
    
    Args:
        preds(torch.Tensor): Predictions from the model
        y(torch.Tensor): True labels

    Returns:
        acc(torch.Tensor): Accuracy of the model
    '''
    predicted = torch.argmax(preds, 1)
    acc = (predicted == y).float().mean()
    return acc


def show_images(dataloader, class_map, cols=4, rows=4, figsize=(8, 8)):
    '''
    Display images from the dataloader

    Args:
        dataloader(torch.utils.data.DataLoader): Dataloader object
        class_map(dict): Dictionary mapping class index to class name like {0: 'cat', 1: 'dog'}
        cols(int): Number of columns in the grid
        rows(int): Number of rows in the grid
        figsize(tuple): Figure size
    '''
    images = next(iter(dataloader))
    fig = plt.figure(figsize=figsize)

    for i in range(0, cols * rows):
        img = images[0][i]
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(torch.permute(img/2+0.5, (1, 2, 0)))
        plt.title(class_map[images[1][i].item()])
        plt.axis('off')


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, figsize=(12, 6)):
    '''
    Plot the loss and accuracy of the model

    Args:
        train_losses(list): List of training losses
        val_losses(list): List of validation losses
        train_accuracies(list): List of training accuracies
        val_accuracies(list): List of validation accuracies
    ''' 
    figure = plt.figure(figsize=figsize)
    cols, rows = 2, 1

    for i, (train_plot, val_plot, label) in enumerate([(train_losses, val_losses, "Loss"), (train_accuracies, val_accuracies, "Accuracy")]):
        figure.add_subplot(rows, cols, i+1)
        plt.plot(train_plot, label=f"Train {label}")
        plt.plot(val_plot, label=f"Val {label}")
        plt.legend()

    plt.show()


def plot_conf_matrix(targets, preds, classes):
    '''
    Plot the confusion matrix

    Args:
        targets(list): List of true labels
        preds(list): List of predicted labels
        classes(list): List of class names
    '''
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes, fmt='d', square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()