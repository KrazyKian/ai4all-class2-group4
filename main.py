import pandas as pd
import numpy as np

from glob import glob # save image files as a list 
import torchvision
import torch
import torchvision.transforms as transforms
import cv2 
import matplotlib.pylab as plt

# Reading in Images
normal = glob('./chest_xray_raw/train/NORMAL/*.jpeg')
disease = glob('./chest_xray_raw/train/PNEUMONIA/*.jpeg')

fig, axs = plt.subplots(1, 2)
axs[0].imshow(cv2.imread(normal[0]))
axs[1].imshow(plt.imread(normal[0]))
axs[0].axis('off')
axs[1].axis('off')
axs[0].set_title('CV Image')
axs[1].set_title('Matplotlib Image')
plt.show()
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