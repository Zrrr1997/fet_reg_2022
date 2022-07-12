# Network architectures
from models.UNet import UNet
from monai.networks.layers import Norm

def prepare_model(device=None, out_channels=None, args=None):

    net = UNet(
        spatial_dims=2,
        in_channels=3, 
        out_channels=out_channels, # softmax output (1 channel per class, i.e. Fg/Bg)
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH
    ).to(device)
    print(f"Using UNet!")
 
    if args.load_weights is not None:
        print('-------\n')
        print(f'Loading weights from {args.load_weights}')
        print('-------')
        net.load_pretrained_unequal(args.load_weights) # ignore layers with size mismatch - needed when changing output channels
    return net
