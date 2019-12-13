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
    global_epoch = 0
    for epoch in tqdm(range(args.num_epochs)):
        if global_epoch == 10:
            args.d_lr /= 10
            args.g_lr /= 10

        for blurred, original in trainloader:
            bs = blurred.shape[0]
            blurred = blurred.to(args.device)
            original = original.to(args.device)
            

            label_fast = torch.empty(bs).to(args.device)
            """ Update D Network """
            d.zero_grad()
            output = d(original).squeeze()
            real_label = label_fast.fill_(1)
            d_error_real = gan_criterion(output, real_label)
            if args.n_gpus > 1:
                d_error_real = d_error_real.mean()
            
            d_error_real.backward()

            fake_data = g(blurred)
            output = d(fake_data.detach()).squeeze()
            fake_label = label_fast.fill_(0)
            d_error_fake = gan_criterion(output, fake_label)
            if args.n_gpus > 1:
                d_error_fake = d_error_fake.mean()
            d_error_fake.backward()

            d_error = d_error_real + d_error_fake
            d_optimizer.step()

            """ Update G Network """
            g.zero_grad()
            real_label = label_fast.fill_(1)
            output = d(fake_data).squeeze()

            g_error = gan_criterion(output, real_label) * 1e-3
            pix_error = pix_criterion(fake_data, original)
            if args.n_gpus > 1:
                g_error = g_error.mean()
                pix_error = pix_error.mean()
            g_cumulative_error = g_error + pix_error
            g_cumulative_error.backward()

            g_optimizer.step()
            
            if global_step % args.print_every == 0:
                print('G loss: {}, D loss: {}'.format(g_cumulative_error.item(), d_error.item()))
                img_idx = random.randint(0, blurred.shape[0] - 1)
                plot_image_comparisons(blurred[img_idx], fake_data[img_idx], original[img_idx])

            if global_step % args.save_every == 0:
                save_model(d, g, global_step, args)
            

            global_step += 1
        global_epoch += 1








