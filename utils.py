import matplotlib.pyplot as plt

def plot_image_comparisons(blurred, generated, original):
    generated = generated.detach().cpu().numpy()
    blurred = blurred.detach().cpu().numpy()
    original = original.detach().cpu().numpy()

    _, axarr = plt.subplots(1,3)
    axarr[0].imshow(blurred)
    axarr[0].set_title('Original')
    axarr[1].imshow(generated)
    axarr[1].set_title('Generated')
    axarr[2].imshow(original)
    axarr[2].set_title('Generated')

    plt.show()

