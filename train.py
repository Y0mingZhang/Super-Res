import torch
from tqdm.auto import tqdm
from utils import plot_image_comparisons, save_model
import random
import torch.nn as nn

def train(d, g, trainloader, args):
    
    d_optimizer = torch.optim.Adam(d.parameters(), args.d_lr)
    g_optimizer = torch.optim.Adam(g.parameters(), args.g_lr)


    gan_criterion = nn.BCEWithLogitsLoss()
    pix_criterion = nn.MSELoss()
    global_step = 0
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
            cumulative_loss = 1e-3 * d_loss + pix_loss
            if args.n_gpus > 1:
                cumulative_loss = cumulative_loss.mean()
            cumulative_loss.backward()
            d.zero_grad()
            
            real_labels = torch.ones(bs).to(args.device)
            fake_labels = torch.zeros(bs).to(args.device)
            d_input = torch.cat((original, generated_imgs_for_d_training))
            d_labels = torch.cat((real_labels, fake_labels))

            d_output = d(d_input).flatten()
            d_loss = gan_criterion(d_output, d_labels)
            if args.n_gpus > 1:
                d_loss = d_loss.mean()
            d_loss.backward()
            
            d_optimizer.step()
            g_optimizer.step()
            d.zero_grad()
            g.zero_grad()
            if global_step % args.print_every == 0:
                print('G loss: {}, D loss: {}'.format(cumulative_loss, d_loss))
                img_idx = random.randint(0, blurred.shape[0] - 1)
                plot_image_comparisons(blurred[img_idx], generated_imgs[img_idx], original[img_idx])

            if global_step % args.save_every == 0:
                save_model(d, g, global_step, args)
        







