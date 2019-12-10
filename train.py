import torch
from tqdm.auto import tqdm
from utils import plot_image_comparisons
import random

def train(d, g, trainloader, args):
    
    d_optimizer = torch.optim.Adam(d.parameters(), args.d_lr)
    g_optimizer = torch.optim.Adam(g.parameters(), args.g_lr)


    gan_criterion = nn.BCEWithLogitsLoss()
    pix_criterion = nn.MSELoss()

    for epoch in tqdm(range(args.num_epochs)):
        for (blurred,_), (original,_) in trainloader:
            bs = blurred.shape[0]
            blurred = blurred.to(args.device)
            original = original.to(args.device)
            
            generated_imgs = g(blurred)
            generated_imgs_for_d_training = generated_imgs.detach()

            preds = d(generated_imgs).flatten()
            

            # Learn G gradients
            # Fake labels
            fake_labels = torch.ones(bs).to(args.device)
            d_loss = gan_criterion(preds, fake_labels)


            # Clean up discriminator grads

            # Do MSELoss
            pix_loss = pix_criterion(generated_imgs, original)

            # Add up losses
            cumulative_loss = d_loss + 1e-3 * pix_loss
            cumulative_loss.backward()
            d.zero_grad()
            
            real_labels = torch.ones(bs).to(args.device)
            fake_labels = torch.zeros(bs).to(args.device)
            d_input = torch.cat((original, generated_imgs_for_d_training))
            d_labels = torch.cat((real_labels, fake_labels))

            d_output = d(d_input).flatten()
            d_loss = gan_criterion(d_output, d_labels)
            d_loss.backward()
            
            d_optimizer.step()
            g_optimizer.step()
            d.zero_grad()
            g.zero_grad()

            print('G loss: {}, D loss: {}'.format(cumulative_loss, d_loss))
            img_idx = random.randint(0, args.batch_size - 1)
            plot_image_comparisons(blurred[img_idx], generated_imgs[img_idx], original[img_idx])







from model import *
from data_pipeline import get_loaders
from argparse import Namespace

args = {
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'd_lr' : 1e-5,
    'g_lr' : 1e-4,
    'num_epochs' : 2,
    'num_resblocks' : 16,
    'overwrite_cache' : False,
    'cache_dir' : 'data_cache/',
    'batch_size' : 32
    
}

args = Namespace(**args)

d = SRGAN_Discriminator(256).to(args.device)
g = SRGAN_Generator(num_resblocks=2).to(args.device)
train_loader, test_loader, val_loader = get_loaders(args)
train(d, g, train_loader, args)


            






