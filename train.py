import torch
from tqdm.auto import tqdm


def train(d, g, trainloader, args):
    
    d_optimizer = torch.optim.Adam(d.parameters(), args.d_lr)
    g_optimizer = torch.optim.Adam(g.parameters(), args.g_lr)


    gan_criterion = nn.BCEWithLogitsLoss
    pix_criterion = nn.MSELoss()

    for epoch in tqdm(range(args.num_epochs)):
        for (blurred,_), (original,_) in trainloader:
            bs = blurred.shape[0]
            blurred = blurred.to(args.device)
            original = original.to(args.device)
            
            generated_imgs = g(blurred)
            generated_imgs_for_d_training = generated_imgs.detach()

            preds = d(generated_imgs)
            

            # Learn G gradients
            # Fake labels
            fake_labels = torch.ones(bs).to(args.device)
            d_loss = gan_criterion(preds, fake_labels)


            # Clean up discriminator grads
            d_loss.backward()
            d.zero_grad()


            # Do MSELoss
            pix_loss = pix_criterion(generated_imgs, original)
            pix_loss.backward()
            
            real_labels = torch.ones(bs).to(args.device)
            fake_labels = torch.zeros(bs).to(args.device)
            d_input = torch.cat((original, generated_imgs_for_d_training))
            d_labels = torch.cat((real_labels, fake_labels))

            d_output = d(d_input)
            d_loss = gan_criterion(d_output, d_input)
            d_loss.backward()
            
            d_optimizer.step()
            g_optimizer.step()






from model import *
from data_pipeline import train_loader
from argparse import Namespace

args = {
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'd_lr' : 1e-3,
    'g_lr' : 1e-3,
    'num_epochs' : 2,
    'num_resblocks' : 16,
    
}

args = Namespace(**args)

d = SRGAN_Discriminator(256)
g = SRGAN_Generator(num_resblocks=2)

train(d, g, train_loader, args)


            






