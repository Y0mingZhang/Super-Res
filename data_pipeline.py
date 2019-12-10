# -*- coding: utf-8 -*-
"""VOC Data Pipeline Test Bench.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RF0mPHNbnRuxZQJzMWhRglPj8pvUwdIr
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil

def get_loaders(args):
    # Use 4x downsampling to approximate blur
    if os.path.isdir(args.cache_dir) and not args.overwrite_cache:
        cache_dir = args.cache_dir

        data_train = torch.load(os.path.join(cache_dir, 'train_dataset.bin'))
        data_test =  torch.load(os.path.join(cache_dir, 'test_dataset.bin'))

    else:
        if os.path.isdir(args.cache_dir):
            shutil.rmtree(args.cache_dir)

        cache_dir = args.cache_dir
        os.mkdir(cache_dir)

        BLURRED_SIZE = 64
        OG_SIZE = 256

        ToTensor = transforms.ToTensor()
        Squeeze = transforms.Lambda(lambda x: x.squeeze())
        blur_transform = transforms.Compose([
                        transforms.CenterCrop(OG_SIZE),
                        transforms.Resize((BLURRED_SIZE,BLURRED_SIZE)),
                        ToTensor,
                        Squeeze
        ])

        og_transform = transforms.Compose([
                        transforms.CenterCrop(OG_SIZE),
                        ToTensor,
                        Squeeze
        ])
        # transforms.Normalize([0.2859], [0.3530]) # Normalize to zero mean and unit variance

        data_train = list(zip(torchvision.datasets.VOCSegmentation('.', download=True, image_set='train', transform=blur_transform, target_transform=blur_transform), 
                        torchvision.datasets.VOCSegmentation('.', download=True, image_set='train', transform=og_transform, target_transform=og_transform)))

        data_test = list(zip(torchvision.datasets.VOCSegmentation('.', download=True, image_set='val', transform=blur_transform, target_transform=blur_transform), 
                        torchvision.datasets.VOCSegmentation('.', download=True, image_set='val', transform=og_transform, target_transform=og_transform)))

        torch.save(data_train, os.path.join(cache_dir, 'train_dataset.bin'))
        torch.save(data_test, os.path.join(cache_dir, 'test_dataset.bin'))

        
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test[:1000], batch_size=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_test[1000:], batch_size=2, shuffle=True)
    
    return train_loader, test_loader, val_loader
