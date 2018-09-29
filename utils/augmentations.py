from albumentations import (
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
    Flip, OneOf, Compose, ToGray, InvertImg
)

from albumentations.core.transforms_interface import DualTransform
import enorm

import numpy as np
import cv2


def convex_coeff(alpha, orig, augmented):
    if alpha == 0:
        return orig
    alpha = np.random.uniform(alpha[0], alpha[1])
    return ((1 - alpha) * orig + (alpha * augmented)).astype(np.uint8)


class Crop(DualTransform):
    """Crops region from image.
    Args:
        window (tuple (int, int)): 
        central (bool): 
        p (float [0, 1]): 
    Targets:
        image, mask
    Image types:
        uint8, float32
    """
    def __init__(self, window, central=False, p=1.0):
        super(Crop, self).__init__(p)
        self.window = np.array(window)
        self.central = central

    def augment_image(self, im, anchors=None):
        if self.central:
            point  = np.array([
                (im.shape[0] - self.window[0]) // 2,
                (im.shape[1] - self.window[1]) // 2
            ])
        elif anchors is not None:
            point = anchors[np.random.randint(len(anchors))] - self.window // 2
            point = np.clip(point, 0, np.array(im.shape[:2]) - self.window)
        else:
            point = np.array([
                np.random.randint(0, max(1, im.shape[0] - self.window[0] + 1)),
                np.random.randint(0, max(1, im.shape[1] - self.window[1] + 1))
            ])

        return im[
            point[0]: point[0] + self.window[0], 
            point[1]: point[1] + self.window[1]
        ]

    def apply(self, img, anchors=None, **params):
        return self.augment_image(img, anchors)


def get_augmentations(strength=1.):
    assert strength > 0, 'Value of `strength` should be of type positive int'
    coeff = int(3 * strength)
    k = max(1, coeff if coeff % 2 else coeff - 1)
    return Compose([
        Compose([
            InvertImg(),
            RandomRotate90(),
            Flip(),
            Transpose(),
        ], p=1.),
        Compose([
            OneOf([
                CLAHE(clip_limit=2, p=.4),
                IAASharpen(p=.3),
                IAAEmboss(p=.3),
            ], p=0.5),
            OneOf([
                IAAAdditiveGaussianNoise(p=.3),
                GaussNoise(p=.7),
            ], p=.5),
            OneOf([
                MotionBlur(p=.2),
                MedianBlur(blur_limit=k, p=.3),
                Blur(blur_limit=k, p=.5),
            ], p=.4),
            OneOf([
                RandomContrast(),
                RandomBrightness(),
            ], p=.4),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=.5, rotate_limit=45, p=.7),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.3),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.6),
        ], p=0.9)
    ])


class Augmentation:
    def __init__(self, side, strength=1.):
        assert strength >= 0, 'Value of `strength` should be of type not negative int'
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        train = bool(strength)
        self.augs = None
        if train:
            self.augs = get_augmentations(strength)

        self.crop = Crop(window=(side + side // 2, side + side // 2), central=not train)
        self.last_crop = Crop(window=(side, side), central=not train)

    def __call__(self, data):
        if data['mask'] is not None:
            data['image'] = np.dstack([data['image'], data['mask']])
        if self.augs is not None:
            data['image'] = self.crop.apply(data['image'])

        data.update({
            'image': data['image'][..., :3],
            'mask': data['image'][..., 3:],
        })
        if self.augs is not None:
            data = self.augs(**data)

        if data['mask'] is not None:
            data['image'] = np.dstack([data['image'], data['mask']])
        data['image'] = self.last_crop.apply(data['image'])

        data.update({
            'image': data['image'][..., :3],
            'mask': data['image'][..., 3:],
        })
        #data.update({'mask': np.expand_dims(data['mask'], -1)})
        return data
