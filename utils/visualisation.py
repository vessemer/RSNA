import matplotlib.pyplot as plt


def show_train_image(image, mask, title=None, axes=True):
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
    ax[0].imshow(image) 
    ax[1].imshow(image)
    ax[1].imshow(mask[..., 0], alpha=0.7, cmap='gray')
    if axes:
        for a in ax:
             a.set_xticks([]); a.set_yticks([])
    if title is not None:
        plt.suptitle(title, fontsize=16)
    plt.show();


def plot_losses(losses):
    _, axes = plt.subplots(ncols=2, figsize=(15, 6))

    for i, key in enumerate(losses.keys()):
            axes[0].plot([v['loss'] for v in losses[key]], label='{} loss'.format(key), alpha=0.7, color='C{}'.format(i))    

    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid()

    for i, key in enumerate(losses.keys()):
        names = list(losses[key][0].keys())
        names.remove('loss')
        for name in names:
            axes[1].plot([v[name] for v in losses[key]], label='{} {}'.format(key, name), alpha=0.7, color='C{}'.format(i))

    axes[1].set_title('Meterics')
    axes[1].legend()
    axes[1].grid()

    plt.show()