import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import time

from math import log10

def set_visible_gpus(args):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(i) for i in args.visible_gpus])  # specify which GPU(s) to be used
    

def tensor_to_img(data):
    img = np.transpose(data.detach().cpu().numpy(), (1, 2, 0))
    img -= img.min()
    img /= img.max()
    return img
def plot_image_comparisons(blurred, generated, original):
    
    _, axarr = plt.subplots(1,3)
    axarr[0].imshow(tensor_to_img(blurred))
    axarr[0].set_title('Blurred')
    axarr[1].imshow(tensor_to_img(generated))
    axarr[1].set_title('Generated')
    axarr[2].imshow(tensor_to_img(original))
    axarr[2].set_title('Original')
    plt.show()

def PSNR(generated, original):
    if original.max() > 2.0:
        print('Assume [0-255] range of pixel intensity')
        amax = 255.0
    else:
        print('Assume [0.0-1.0] range of pixel intensity')
        amax = 1.0
    
    

    generated = np.clip(generated, amin=0.0, amax=amax)
    original = np.clip(generated, amin=0.0, amax=amax)
    # Compute MSE
    mse = ((generated - original) ** 2).sum()
    
    return 20 * np.log10(amin) - 10 * np.log10(mse)




def save_model(d, g, step, args):
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    torch.save(g, os.path.join(args.model_dir, 'G@{}.bin'.format(step)))
    # Realized no need to save D's
    #torch.save(d, os.path.join(args.model_dir, 'D@{}.bin'.format(step)))
