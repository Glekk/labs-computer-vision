from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from models import CNN, SECNN, DepthwiseSeparableCNN, ResBottleneckCNN


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


def show_images(dataloader, class_map, cols=4, rows=4):
    '''
    Display images from the dataloader

    Args:
        dataloader(torch.utils.data.DataLoader): Dataloader object
        class_map(dict): Dictionary mapping class index to class name like {0: 'cat', 1: 'dog'}
        cols(int): Number of columns in the grid
        rows(int): Number of rows in the grid
    '''
    images_sample = next(iter(dataloader))

    fig = plt.figure(figsize=(8, 8))

    for i in range(cols*rows):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(images_sample[0][i].permute(1, 2, 0)/2 + 0.5)
        plt.title(class_map[images_sample[1][i].item()])
        plt.axis('off')
    plt.show()


def get_models(image_channels, classes, device):
    '''
    Get models for the given image channels and classes

    Args:
        image_channels(int): Number of channels in the image
        classes(list): List of class names
        device(torch.device): Device for the models

    Returns:
        standard_cnn(torch.nn.Module): Standard CNN model
        se_cnn(torch.nn.Module): SE CNN model
        depthwise_cnn(torch.nn.Module): Depthwise Separable CNN model
        resbottleneck_cnn(torch.nn.Module): ResBottleneck CNN model
    '''
    standard_cnn = CNN(image_channels, len(classes)).to(device)
    se_cnn = SECNN(image_channels, len(classes)).to(device)
    depthwise_cnn = DepthwiseSeparableCNN(image_channels, len(classes)).to(device)
    resbottleneck_cnn= ResBottleneckCNN(image_channels, len(classes)).to(device)

    return standard_cnn, se_cnn, depthwise_cnn, resbottleneck_cnn


def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, name, figsize=(8, 4)):
    '''
    Plot training and validation metrics

    Args:
        train_losses(list): List of training losses
        train_accuracies(list): List of training accuracies
        val_losses(list): List of validation losses
        val_accuracies(list): List of validation accuracies
        name(str): Name of the model
        figsize(tuple): Figure size
    '''
    fig = plt.figure(figsize=figsize)
    cols, rows = 2, 1
    for i, (train, val) in enumerate([(train_losses, val_losses), (train_accuracies, val_accuracies)]):
        fig.add_subplot(rows, cols, i+1)
        plt.plot(train, label='Training')
        plt.plot(val, label='Validation')
        plt.legend()

        plt.title(f'{name} loss' if i == 0 else f'{name} accuracy')
    plt.show()


def get_acc(predicitons, labels):
    '''
    Get accuracy of the model

    Args:
        predicitons(torch.Tensor): Model predictions
        labels(torch.Tensor): True labels

    Returns:
        torch.Tensor: Accuracy of the model
    '''
    return (predicitons.argmax(dim=1) == labels).float().mean()