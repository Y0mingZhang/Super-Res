import matplotlib.pyplot as plt

def plot_image_comparisons(blurred, original, generated):
    generated = generated.detach().cpu().numpy()
    blurred = blurred.cpu().numpy()
    original = original.cpu().numpy()

    _, axarr = plt.subplots(1,3)
    axarr[0,0].imshow(blurred)
    axarr[0,0].title('Original')
    axarr[0,1].imshow(generated)
    axarr[0,1].title('Generated')
    axarr[0,2].imshow(original)
    axarr[0,2].title('Generated')

    plt.show()