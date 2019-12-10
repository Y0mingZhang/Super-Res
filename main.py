from model import *
from data_pipeline import get_loaders
from train import train
from argparse import Namespace
from utils import set_visible_gpus

args = {
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'd_lr' : 1e-5,
    'g_lr' : 1e-4,
    'num_epochs' : 5,
    'num_resblocks' : 16,
    'overwrite_cache' : False,
    'cache_dir' : 'data_cache/',
    'batch_size' : 48,
    'print_every' : 100,
    'save_every' : 2000,
    'visible_gpus' : [1,0,4,5],
    'model_dir' : 'model_cache/'
    
}

args = Namespace(**args)

if args.visible_gpus:
    set_visible_gpus(args)

g = SRGAN_Generator().to(args.device)
d = SRGAN_Discriminator(256)

train_loader, test_loader, val_loader = get_loaders(args)
train(d, g, train_loader, test_loader, args)
