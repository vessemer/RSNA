"""
This is an implementation of localized energy-based normalization from paper proposed by:
R. H. H. M. Philipsen et al,
"Localized Energy-Based Normalization of Medical Images: Application to Chest Radiography" 
in IEEE Transactions on Medical Imaging.
doi: 10.1109/TMI.2015.2418031
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7073580&isnumber=7229384
"""

from scipy.ndimage import filters
import numpy as np


class EnergyNormalization:
    def __init__(self, B=7, alpha=2, gamma=2):
        self.B = B
        self.alpha = alpha
        self.gamma = gamma

    def decompose(self, images):
        decomposed = []
        blured = [images]
        for i in range(self.B):
            blured += [[filters.gaussian_filter(img, self.alpha * self.gamma ** i) for img in images]]
            decomposed += [[old  - new
                           for old, new in zip(blured[i], blured[i + 1])]]
        decomposed += [blured[-1]]
        return list(zip(*decomposed))

    def get_energies(self, images, masks):
        energies = list()
        for img, mask in zip(images, masks):
            energies.append([
                np.std(layer[mask])
                for layer in img
            ])
        return energies

    def normalize(self, decomposed, masks, immutable=[], immutable_masks=[]):
        energies = self.get_energies(decomposed, masks)
        immutable_energies = self.get_energies(immutable, immutable_masks)

        referenced = np.asarray([
            e.mean() 
            for e in np.asarray(immutable_energies).T
        ])

        images = []
        diff = []
        for imgs, energy in zip(decomposed, energies):
            normalized = np.zeros(imgs[0].shape)
            for layer, e, ref in zip(imgs, energy, referenced):
                diff += [ref / e]
                normalized += layer * diff[-1]
            images += [normalized]

        return images, diff


    def iterate_normalization(self, images, masks, 
                              immutable=[], immutable_masks=[], 
                              n_iterations=10, verbose=True):
        decomposed_immutable = self.decompose(immutable)
        for i in range(n_iterations):
            decomposed = self.decompose(images)
            images, diff = self.normalize(decomposed, masks, decomposed_immutable, immutable_masks) 
            if verbose:
                print('Step: ', i, ', diff: ', np.mean(diff))
        return images
