import pandas as pd
import nibabel as nib
import numpy as np
import torch
import os
import cv2
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from monai.utils import set_determinism
from monai.data import DataLoader, Dataset, CacheDataset
from monai.data import list_data_collate, decollate_batch
from monai.transforms import Resize, AddChannel

from utils.transforms import prepare_transforms

def read_fns(in_dir='./fetoscopy-placenta-dataset/Fetoscopy Placenta Dataset/Vessel_segmentation_annotations', debug=False, task='segmentation'):
 
    
    set_determinism(seed=0)
    print(f"Input directory: {in_dir}")

    paths_images = glob(f'{in_dir}/**/images/*.png', recursive=True)
    paths_segs = glob(f'{in_dir}/**/masks_gt/*.png', recursive=True)
    train_images = sorted([el for el in paths_images if 'video01' not in el])
    val_images = sorted([el for el in paths_images if 'video01' in el])
    train_segs = sorted([el for el in paths_segs if 'video01' not in el])
    val_segs = sorted([el for el in paths_segs if 'video01' in el])
    if task == 'reconstruction':
        train_segs = train_segs + val_segs
        train_images = train_images + val_images
        val_images = train_images
        val_segs = train_segs

    train_files = [{"image": image, "seg": seg, "fn": image} for image, seg in zip(train_images, train_segs)]
    val_files = [{"image": image, "seg": seg, "fn": image} for image, seg in zip(val_images, val_segs)]
    print("Train:", len(train_files), "Val:", len(val_files))
    if debug: 
        train_files = train_files[:4]
        val_files = val_files[:4]

    return train_files, val_files

def prepare_loaders(spatial_size=[224, 224], cache=True, batch_size=4, debug=False, task='segmentation'):
 
    train_files, val_files = read_fns(debug=debug, task=task)

    
    train_transforms, val_transforms = prepare_transforms(spatial_size=spatial_size)

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size, 
                              num_workers=0, # must be 0 to avoid "unexpected exception"
                              collate_fn=list_data_collate, 
                              pin_memory=torch.cuda.is_available()
                              )
    val_loader = DataLoader(val_ds,
                              batch_size=batch_size, 
                              num_workers=0, 
                              collate_fn=list_data_collate, 
                              pin_memory=torch.cuda.is_available()
                              )

    return train_loader, val_loader
