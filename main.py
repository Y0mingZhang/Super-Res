
from model import *
from data_pipeline import get_loaders
from train import train
from argparse import Namespace
from utils import set_visible_gpus

args = {
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'd_lr' : 1e-4,
    'g_lr' : 1e-4,
    'num_epochs' : 15,
    'num_resblocks' : 16,
    'overwrite_cache' : True,
    'cache_dir' : 'data_cache/',
    'batch_size' : 64,
    'print_every' : 4,
    'save_every' : 23,
    'visible_gpus' : [1,0,4,5],
    'model_dir' : 'model_cache/'
    
}

args = Namespace(**args)

# if torch.cuda.device_count() > 1:
#     args.n_gpus = torch.cuda.device_count()
# else:
#     args.n_gpus = 0
# ''
# g = SRGAN_Generator().to(args.device)
# d = SRGAN_Discriminator(256).to(args.device)
# if args.n_gpus > 1:
#     g = nn.DataParallel(g)
#     d = nn.DataParallel(d)


train_loader, test_loader, val_loader = get_loaders(args)

train_loader.datasets[0][0][0]

# train(d, g, train_loader, args)