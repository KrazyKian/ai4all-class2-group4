import pandas as pd
import numpy as np

from glob import glob # save image files as a list 

import cv2 
import matplotlib.pylab as plt

# Reading in Images
normal = glob('./chest_xray_raw/train/NORMAL/*.jpeg')
disease = glob('./chest_xray_raw/train/PNEUMONIA/*.jpeg')

# display an image with cv vs matplot libararies
fig, axs = plt.subplots(1, 2)
axs[0].imshow(cv2.imread(normal[0]))
axs[1].imshow(plt.imread(normal[0]))
axs[0].axis('off')
axs[1].axis('off')
axs[0].set_title('CV Image')
axs[1].set_title('Matplotlib Image')
plt.show()
# Note: cv2 read in image data correctly - matplotlib read in has weird color when displayed