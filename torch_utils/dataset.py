import os
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose


MEAN = [0.50600299]
STD = [0.20282886]

img_transform = Compose([
    ToTensor(),
    Normalize(mean=MEAN, std=STD)
])


def get_paths(df, root_dir, seed=None):
    keys = pd.Series(np.sort(pd.unique(df.ImageId)))
    if seed is not None:
        rs = np.random.RandomState(seed=seed)
        rs.shuffle(keys.values)
    keys = keys.values.tolist()

    paths = dict()
    for i, key in enumerate(keys):
        paths[key] = { 
            'image': os.path.join(root_dir, 'png', keys[i]),
            'inversed': os.path.join(root_dir, 'inversed', '{}', keys[i]),
            'mask': os.path.join(root_dir, 'masks', keys[i])
        }

    return paths


def get_fold(pths, fold=0, nb_folds=1):
    """
    Arguments:
        pths dict{str: dict}
        fold int: Fold to return. Should be positive.
        nb_folds: Overall amount of folds to split on.
            Should be greater than `fold`.

    Returns:
        tuple (dict, dict): fold paths and rest paths dicts correspondingly.
    """
    assert fold >= 0, 'Value of `fold` should be of type non negative int'
    assert fold < nb_folds, 'Value of `fold` should be less than `nb_folds`'
    keys = sorted(pths.keys())
    step = len(keys) // nb_folds

    keys_fold = keys[fold * step : (fold + 1) * step]
    keys_rest = keys[:fold * step] + keys[(fold + 1) * step:]

    return (
        {k: v for k, v in pths.items() if k in keys_fold}, 
        {k: v for k, v in pths.items() if k in keys_rest}
    )


class CXR_Dataset(Dataset):
    def __init__(self, paths, augmentations=None, inverse=False):
        self.transform = augmentations
        self.paths = paths.copy()
        self.inverse = inverse
        self.keys = list(self.paths.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        meta = self.paths[self.keys[idx]]
        im_path = meta['image']
        if np.random.randint(1 + self.inverse) and ('inversed' in meta.keys()):
            inverse_id = np.random.randint(self.inverse) + 1
            im_path = os.path.join(meta['inversed'].format(inverse_id))
        image = cv2.imread(im_path)
        mask = cv2.imread(meta['mask'], 0)

        if mask is None:
            mask = np.zeros(image.shape[:-1], dtype=np.uint8)
        mask = np.expand_dims(mask, -1).astype(np.uint8)

        data = {"image": image, "mask": mask}
        if self.transform is not None:
            augmented = self.transform(data)
            image, mask = augmented["image"], augmented["mask"]

        mask = (mask > 120).astype(np.float32)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, -1)
            
        return { 
            'image': img_transform(np.expand_dims(np.mean(image, -1), -1).astype(np.uint8)),
#             'image': img_transform(image),
            'mask': torch.from_numpy(np.rollaxis(mask, 2, 0)),
        }
