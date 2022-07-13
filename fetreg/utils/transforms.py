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
    RandRotated,
    RandAffined,
    RandSpatialCropd,
    RandShiftIntensityd,
    RandFlipd,
    Compose
)
import numpy as np
def prepare_transforms(spatial_size=[224, 224]):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "seg"]),
            AsChannelFirstd(keys=["image", "seg"]),
            RandAffined(keys=["image", "seg"], translate_range=(44, 44), rotate_range=(np.pi/3, np.pi/3), scale_range=(1.1, 1.1), mode=("bilinear", "nearest")),
            RandFlipd(keys=["image", "seg"], spatial_axis=0),
            RandFlipd(keys=["image", "seg"], spatial_axis=1),
            RandSpatialCropd(keys=["image", "seg"], roi_size=[224, 224]),
            RandShiftIntensityd(keys=["image"], offsets=51),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1), 
            Resized(keys=["image"], spatial_size=spatial_size),
            Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
            ToTensord(keys=["image", "seg"]), 
        ]
    )
    val_transforms= Compose(
        [
            LoadImaged(keys=["image", "seg"]),
            AsChannelFirstd(keys=["image", "seg"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1), 
            Resized(keys=["image"], spatial_size=spatial_size),
            Resized(keys=["seg"], spatial_size=spatial_size, mode="nearest-exact"),
            ToTensord(keys=["image", "seg"]), 
        ]
    )
    return train_transforms, val_transforms
