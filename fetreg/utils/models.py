# Network architectures
from models.UNet import UNet
from models.Nested_UNet import NestedUNet
from monai.networks.layers import Norm
import segmentation_models_pytorch as smp

import torch

def prepare_model(device=None, out_channels=None, args=None):

    if args.arch == 'unet':
        net = UNet(
            spatial_dims=2,
            in_channels=3, 
            out_channels=out_channels, # softmax output (1 channel per class, i.e. Fg/Bg)
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=4,
            norm=Norm.BATCH
        ).to(device)
        print(f"Using UNet!")
    elif args.arch == 'unet++':
        '''
        net = NestedUNet(
           num_classes=2,
           input_channels=3
        ).to(device)
        '''
        net = smp.Unet('resnet101', classes=2, activation='softmax', encoder_weights='imagenet').to(device)
 
        print(f"Using UNet++!")
 
    if args.load_weights is not None:
        print('-------\n')
        print(f'Loading weights from {args.load_weights}')
        print('-------')
        if args.arch == 'unet++':
            net.load_state_dict(torch.load(args.load_weights)['net'])
        else:
            net.load_pretrained_unequal(args.load_weights) # ignore layers with size mismatch - needed when changing output channels
    return net
