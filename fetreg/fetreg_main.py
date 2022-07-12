from tqdm import tqdm
import numpy as np
import os
import sys
import logging
import argparse
from sklearn.metrics import confusion_matrix

# Torch
import torch
import torch.nn.functional as nnf
from torch.utils.tensorboard import SummaryWriter



# IGNITE
import ignite
from ignite.engine import (
    Events,
    _prepare_batch,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, ConfusionMatrix, mIoU

# MONAI
import monai
from monai.transforms import (
    AsDiscrete,
    Identity
)
from monai.utils import first
from monai.data import list_data_collate, decollate_batch
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import compute_meandice
from monai.handlers import (
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    LrScheduleHandler,
    MeanSquaredError,
)

# Utils
from utils.data_utils import prepare_loaders
from utils.parser import prepare_parser
from utils.transforms import prepare_transforms
from utils.models import prepare_model 



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser(description='FetReg Challenge.')
    parser = prepare_parser(parser)
    args = parser.parse_args()
    print('--------\n')
    print(args, '\n')
    print('--------')


    out_channels = 2 

    # Create Model, Loss, and Optimizer
    device = torch.device(f"cuda:{args.gpu}")
    net = prepare_model(device=device, out_channels=out_channels, args=args)

    
    train_loader, val_loader = prepare_loaders(batch_size=args.batch_size, 
                                       debug=args.debug
                                       )
    writer = SummaryWriter(log_dir = args.log_dir)
    check_data = first(train_loader)
    print('Input image shape check:', check_data["image"].shape, 'Segmentation mask shape check:', check_data["seg"].shape)



       
    # Hyperparameters
    loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True, jaccard=True) 
    lr = args.lr
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-5)
 
    def prepare_batch(batch, device=None, non_blocking=False):
        imgs = batch["image"]
        segs = batch["seg"]
        segs = (segs > 0) * 1 # Binarize
        segs, _ = torch.max(segs, dim=1) # Collapse to 1 channel
        segs = segs.unsqueeze(dim=1)
        return _prepare_batch((imgs, segs), device, non_blocking)

    trainer = create_supervised_trainer(
        net, opt, loss, device, False, prepare_batch=prepare_batch
    )
    checkpoint_handler = ModelCheckpoint(
        args.ckpt_dir, "net", n_saved=20, require_empty=False
    )
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED(every=args.save_every),
        handler=checkpoint_handler,
        to_save={"net": net, "opt": opt},
    )

    # Logging
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch
    train_tensorboard_stats_handler = TensorBoardStatsHandler(output_transform=lambda x: x, log_dir=args.log_dir,)
    train_tensorboard_stats_handler.attach(trainer)

    # Learning rate drop-off at every args.lr_step_size epochs
    train_lr_handler = LrScheduleHandler(lr_scheduler=torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=0.1), print_lr=True) 
    train_lr_handler.attach(trainer)

    # Validation configuration
    validation_every_n_iters = args.eval_every
    def binary_one_hot_output_transform(output):
        y_pred, y = output
        
        y_pred = torch.cat(y_pred).flatten()
        y = torch.cat(y).flatten()
        #y_pred = torch.sigmoid(y_pred).round().long()
        y_pred = ignite.utils.to_onehot(y_pred.round().long(), 2)
        y = y.long()
        return y_pred, y
    metric_name = "Mean_Dice"
    m_IoU = mIoU(ConfusionMatrix(num_classes=2, output_transform=binary_one_hot_output_transform))
    val_metrics = {metric_name: MeanDice(), 'mIoU': m_IoU}

    num_classes = 2

    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    evaluator = create_supervised_evaluator(
        net,
        val_metrics,
        device,
        True,
        output_transform=lambda x, y, y_pred: ([post_pred(i) for i in decollate_batch(y_pred)], [post_label(i) for i in decollate_batch(y)]),
        prepare_batch=prepare_batch,
    )



    @trainer.on(Events.ITERATION_COMPLETED(every=validation_every_n_iters))
    def run_validation(engine):
        evaluator.run(val_loader)


    @trainer.on(Events.ITERATION_COMPLETED(every=validation_every_n_iters))
    def save_images(engine):

        with torch.no_grad():   
            x = first(val_loader)['image']
            seg = first(val_loader)['seg']
            # TODO save images with tensorboard handler

    

    # Stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None, 
        global_epoch_transform=lambda x: trainer.state.epoch,
    ) 
    val_stats_handler.attach(evaluator)

    # Handler to record metrics to TensorBoard at every validation epoch
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.iteration,
        log_dir=args.log_dir,
    )  # fetch global iteration number from trainer
    val_tensorboard_stats_handler.attach(evaluator)


    train_epochs = args.epochs
    state = trainer.run(train_loader, train_epochs)
    print(state)




    
