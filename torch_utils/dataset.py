import os
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose


MEAN = [0.50600299]
STD = [0.20282886]

img_transform = Compose([
    ToTensor(),
    Normalize(mean=MEAN, std=STD)
])


def get_paths(df, root_dir, mask_dir='masks'):
    keys = pd.Series(np.sort(pd.unique(df.ImageId)))
    keys = keys.values.tolist()

    paths = dict()
    for i, key in enumerate(keys):
        paths[key] = { 
            'image': os.path.join(root_dir, 'png', keys[i]),
            'inversed': os.path.join(root_dir, 'inversed', '{}', keys[i]),
            'mask': os.path.join(root_dir, mask_dir, keys[i])
        }

    return paths


def get_fold(pths, fold=0, nb_folds=1, seed=None):
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
    if seed is not None:
        rs = np.random.RandomState(seed=seed)
        rs.shuffle(keys)

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
            inv_path = os.path.join(meta['inversed'].format(inverse_id))
            if os.path.isfile(inv_path):
                im_path = inv_path
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
            'mask': torch.from_numpy(np.rollaxis(mask, 2, 0)),
        }


class BBoxDataset(Dataset):
    """Coco dataset."""

    def __init__(self, paths, annotations, class_df, augmentations=None, inverse=False):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = augmentations
        self.paths = paths.copy()
        self.inverse = inverse
        self.annotations = annotations
        self.class_df = class_df
        self.keys = list(self.paths.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        meta = self.paths[key]
        image = self.load_image(meta)
        bboxes = self.load_annotations(key)
        label = self.load_label(key)
        data = {"image": image, "bboxes": bboxes, 'category_id': [1] * len(bboxes)}
        if self.transform is not None:
            augmented = self.transform(data)
            image, bboxes = augmented["image"], np.array(augmented["bboxes"])

        if len(bboxes) == 0:
            bboxes = np.zeros((0, 4))
        else:
            bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
            bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        bboxes = np.pad(bboxes, [[0, 0], [0, 1]], mode='constant', constant_values=1)

        return self.postprocess(image, bboxes, key, label)

    def postprocess(self, image, bboxes, key, label):
        return { 
            'img': img_transform(np.expand_dims(np.mean(image, -1), -1).astype(np.uint8)),
            'annot': bboxes.astype(np.int),
            'pid': key,
            'label': label.astype(np.float32),
        }

    def load_image(self, meta):
        im_path = meta['image']
        if np.random.randint(1 + self.inverse) and ('inversed' in meta.keys()):
            inverse_id = np.random.randint(self.inverse) + 1
            inv_path = os.path.join(meta['inversed'].format(inverse_id))
            if os.path.isfile(inv_path):
                im_path = inv_path

        return cv2.imread(im_path)

    def load_annotations(self, key):
        # get ground truth annotations
        key = key.split('.')[0]
        annotation = self.annotations.query('patientId==@key')
        annotation = annotation.drop(['patientId'], axis=1).values
        bboxes = np.zeros((0, 4))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation) == 0:
            return bboxes

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
#         annotation[:, 2] = annotation[:, 0] + annotation[:, 2]
#         annotation[:, 3] = annotation[:, 1] + annotation[:, 3]

        return annotation

    def load_label(self, key):
        key = key.split('.')[0]
        label = self.class_df.query('patientId==@key')
        label = label['class'].values[0]
        l = np.zeros((3, ))
        l[label] = 1
        return l

    def num_classes(self):
        return 1    


def bbox_collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    pids = [s['pid'] for s in data]
    labels = [s['label'] for s in data]
    # scales = [s['scale'] for s in data]
    
    labels = torch.tensor(np.asarray(labels))
        
    widths = [int(s.shape[1]) for s in imgs]
    heights = [int(s.shape[2]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, 1, max_width, max_height)

    for i in range(batch_size):
        padded_imgs[i, :, :int(imgs[i].shape[1]), :int(imgs[i].shape[2])] = imgs[i]

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = torch.tensor(annot)
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    return {'img': padded_imgs, 'annot': annot_padded, 'pid': pids, 'label': labels}


class EqualizedSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, class_df, max_negatives=None):
        self.data_source = data_source
        keys = [k.split('.')[0] for k in data_source.keys]
        self.keys = keys
        query = class_df.query('patientId==@keys').reset_index(drop=True)
        self.groups = query.groupby(['class']).groups.values()
        self.groups = [np.array(v) for v in self.groups]
        self.min_class_samples = [min([len(v) for v in self.groups])] * len(self.groups)
        if max_negatives is not None:
            self.min_class_samples = [ 
                max_negatives,
                len(self.groups[1]),
                max_negatives,
            ]

    def __iter__(self):
        for v in self.groups:
            np.random.shuffle(v)
        idxs = np.concatenate([v[:self.min_class_samples[i]] for i, v in enumerate(self.groups)])
        np.random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return sum(self.min_class_samples)


class ValSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.keys = data_source.keys.copy()

    def __iter__(self):
        return iter(np.arange(len(self.keys)))

    def __len__(self):
        return len(self.data_source)
