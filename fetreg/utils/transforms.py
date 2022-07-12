from monai.transforms import(
    AddChanneld,
    AsChannelFirstd,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    ConcatItemsd,
    RandCropByPosNegLabeld,
    Compose
)
def prepare_transforms(spatial_size=[448, 448]):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "seg"]),
            #AddChanneld(keys=["image", "seg"]),
            AsChannelFirstd(keys=["image", "seg"]),
            Resized(keys=["image"], spatial_size=spatial_size),
            Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
            ToTensord(keys=["image", "seg"]), 
        ]
    )
    val_transforms= Compose(
        [
            LoadImaged(keys=["image", "seg"]),
            #AddChanneld(keys=["image", "seg"]),
            AsChannelFirstd(keys=["image", "seg"]),
            Resized(keys=["image"], spatial_size=spatial_size),
            Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
            ToTensord(keys=["image", "seg"]), 
        ]
    )
    return train_transforms, val_transforms
