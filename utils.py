import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def set_visible_gpus(args):
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(i) for i in args.visible_gpus])  # specify which GPU(s) to be used
    

def tensor_to_img(data):
    return np.transpose(data.detach().cpu().numpy(), (1, 2, 0))

def plot_image_comparisons(blurred, generated, original):
    
    _, axarr = plt.subplots(1,3)
    axarr[0].imshow(tensor_to_img(blurred))
    axarr[0].set_title('Blurred')
    axarr[1].imshow(tensor_to_img(generated))
    axarr[1].set_title('Generated')
    axarr[2].imshow(original)
    axarr[2].set_title('Original')

    plt.show()


def save_model(model, step, args):
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    torch.save(model, os.path.join(args.model_dir, '@{}.bin'.format(step)))
