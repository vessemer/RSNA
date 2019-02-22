from albumentations import (
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, 
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
    Flip, OneOf, Compose, ToGray, InvertImg, HorizontalFlip
)

from albumentations.core.transforms_interface import DualTransform
import imgaug as ia
from imgaug import augmenters as iaa
import enorm

import numpy as np
import cv2


def convex_coeff(alpha, orig, augmented):
    if alpha == 0:
        return orig
    alpha = np.random.uniform(alpha[0], alpha[1])
    return ((1 - alpha) * orig + (alpha * augmented)).astype(np.uint8)


class BBoxesAffine(DualTransform):
    def __init__(self, translate_percent=(-.1, .1), rotate=(-30, 30), scale=(0.6, 1.2), p=1.0):
        super(BBoxesAffine, self).__init__(p)
        self.seq = iaa.Sequential([
            iaa.Affine(
                translate_percent=translate_percent, 
                rotate=rotate, 
                scale=scale
            )
        ])

    def augment_image(self, data):
        bbxs = list()
        data['bboxes'][:, 2] += data['bboxes'][:, 0]
        data['bboxes'][:, 3] += data['bboxes'][:, 1]
        for bbx in data['bboxes']:
            bbxs.append(ia.BoundingBox(*bbx.tolist()))
        bbxs = [ia.BoundingBoxesOnImage(bbxs, data['image'].shape[:2])]

        seq_det = self.seq.to_deterministic()
        data['image'] = seq_det.augment_images(data['image'][np.newaxis])[0]
        bbx_aug = seq_det.augment_bounding_boxes(bbxs)[0]
        bbx_aug = bbx_aug.cut_out_of_image()

        bbxs = list()
        for bbx in bbx_aug.bounding_boxes:
            bbxs.append([bbx.x1_int, bbx.y1_int, int(bbx.width), int(bbx.height)])
        data['bboxes'] = np.array(bbxs)

        return data

    def apply(self, data, **params):
        return self.augment_image(data)


def get_bboxes_augmentations(strength=1.):
    assert strength > 0, 'Value of `strength` should be of type positive int'
    coeff = int(3 * strength)
    k = max(1, coeff if coeff % 2 else coeff - 1)
    return Compose([
        Compose([
#             InvertImg(),
#             RandomRotate90(),
#             Flip(),
#             Transpose(),
            HorizontalFlip(),
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
#             ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-.5, 0.), rotate_limit=45, p=.7),
#             OneOf([
#                 OpticalDistortion(p=0.3),
#                 GridDistortion(p=0.3),
#                 IAAPiecewiseAffine(p=0.3),
#             ], p=0.6),
        ], p=0.9)
    ], bbox_params={'format': 'coco', 'min_area': 22, 'min_visibility': .1, 'label_fields': ['category_id']})


class BBoxesAugmentation:
    def __init__(self, side, strength=1.):
        assert strength >= 0, 'Value of `strength` should be of type not negative int'
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        train = bool(strength)
        self.augs = None
        if train:
            self.augs = get_bboxes_augmentations(strength)

        self.affine = BBoxesAffine()

    def __call__(self, data):
        if self.augs is not None:
            data = self.affine.apply(data)
            data = self.augs(**data)

        return data
