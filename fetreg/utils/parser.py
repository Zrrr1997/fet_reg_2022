import argparse

def prepare_parser(parser):
    parser.add_argument('--lr', type=float, default=1e-4,
                         help='Learning rate.')
    parser.add_argument('--lr_step_size', type=int, default=1000,
                         help='Decrease learning rate every lr_step_size epochs.')
    parser.add_argument('--gpu', type=int, default=1,
                         help='GPU device index.')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/test',
                         help='Checkpoint directory.')
    parser.add_argument('--batch_size', type=int, default=16,
                         help='Batch size for data loaders.')
    parser.add_argument('--epochs', type=int, default=1000,
                         help='Training epochs.')
    parser.add_argument('--log_dir', type=str, default='./runs/test',
                         help='Logs directory.')
    parser.add_argument('--save_every', type=int, default=115,
                         help='Number of epochs to save a model.')
    parser.add_argument('--eval_every', type=int, default=115,
                         help='Number of iterations to evaluate model.')
    parser.add_argument('--load_weights', type=str, default=None,
                         help='Load model from this directory.')
    parser.add_argument('--debug', default=False, action='store_true',
                         help='Debug with one sample for training and validation.')
    return parser
