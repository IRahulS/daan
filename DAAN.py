import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
from datetime import datetime
import math
import scipy.io as sio
from scipy import integrate
import cv2 as cv
import torch
import torch.nn.functional as F
import demo_real_data_Moffett Field data
import demo_synthetic_data

# run Moffett Field data
if __name__ == '__main__':
    im = np.load('/data/Moffett Field/Y.npy')
    edm = np.load('/data/Moffett Field/E.npy')
    model = AutoUnmix(L=189, P=3, Q=0.8, Z=1, init_edm=edm.T, height=50, width=50)
    abd = model.fit(im.reshape(-1, 189).T, max_iter=1000, verbose=True).T
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(np.reshape(abd[:,i], (50,50)))
        plt.colorbar(fraction=0.05)
        plt.axis('off')
    plt.show()

 # run Synthetic data
# if __name__ == '__main__':
#     im = np.load('/data/Synthetic data/Y.npy')
#     edm = np.load('/data/Synthetic data/E.npy')
#     model = AutoUnmix(L=224, P=3, Q=0.8, Z=1, init_edm=edm.T, height=26, width=26)
#     abd = model.fit(im.reshape(-1, 224).T, max_iter=1000, verbose=True).T
#     for i in range(3):
#         plt.subplot(1, 3, i+1)
#         plt.imshow(np.reshape(abd[:,i], (26,26)))
#         plt.colorbar(fraction=0.05)
#         plt.axis('off')
#     plt.show()