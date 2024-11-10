import pandas as pd
import numpy as np

from glob import glob # save image files as a list 
import torchvision
import torch
import torchvision.transforms as transforms
import cv2 
import matplotlib.pyplot as plt
import random

# Reading in Images
normal = glob('./chest_xray_raw/train/NORMAL/*.jpeg')
disease = glob('./chest_xray_raw/train/PNEUMONIA/*.jpeg')
normal = random.sample(normal, 500)
disease = random.sample(normal, 500)

# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(cv2.imread(normal[0]))
# axs[1].imshow(plt.imread(normal[0]))
# axs[0].axis('off')
# axs[1].axis('off')
# axs[0].set_title('CV Image')
# axs[1].set_title('Matplotlib Image')
# plt.show()
# Note: cv2 read in image channels/colors correctly, kept x-ray in black and white (cv2 uses BGR, data is in BGR)

# Preprocessing
def get_mean_and_std(loader):
    mean = 0
    std = 0
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
    
    mean /= total_images_count
    std /= total_images_count
    return mean, std

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # size will be dependent on deep learning model used & which size gives best performance

    # randomly alter images to prepare model for any possible x-ray in test set! 
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10),

    transforms.ToTensor(), # converting image data to PyTorch tensor format (0.0-1.0) expected by Pytorch models tensor (this is a multi-dimensional array data structure)
])

train_dataset = torchvision.datasets.ImageFolder(root = './chest_xray_raw/train', transform = train_transforms)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True)
mean, std = get_mean_and_std(train_loader)   

    # updating transforms after normalizing (normalization stabilizes and accelerates training process, adjusting data to a mean of 0 and std of 1. results in faster convergence, optimizes model training process)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10),
    transforms.ToTensor(), 
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# Preprocessed Image Lists
train_dataset = torchvision.datasets.ImageFolder(root = './chest_xray_raw/train', transform = train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root = './chest_xray_raw/test', transform = test_transforms)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 32, shuffle = True)

# Training Neural Network
def set_device(): 
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)

def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()

    for epoch in range(n_epochs):
        print("Epoch number %d " % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * running_correct / total
        print("Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f" % (running_correct, total, epoch_acc, epoch_loss))
        
        evaluate_model_on_test_set(model, test_loader)

    print("Finished")
    return model

def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    running_correct = 0.0
    total = 0
    device = set_device()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

    epoch_acc = 100.0 * running_correct / total
    print("Testing dataset. Got %d out of %d images correctly (%.3f%%)" % (running_correct, total, epoch_acc))

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# input parameters for train_nn
resnet18_model = models.resnet18(pretrained = True)
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 2 # pneumonia or no pneumonia 
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
device = set_device()
resnet18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss() # shows correctness of model classification
optimizer = optim.SGD(resnet18_model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.003)

train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, 1)