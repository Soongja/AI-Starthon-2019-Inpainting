import random
import numpy as np
import torch
import cv2
from albumentations import (
    OneOf, Compose, HorizontalFlip, ShiftScaleRotate, GridDistortion, ElasticTransform,
    RandomGamma, RandomContrast, RandomBrightness, RandomBrightnessContrast,
    Blur, MedianBlur, MotionBlur,
    CLAHE, IAASharpen, GaussNoise, IAAAdditiveGaussianNoise,
    HueSaturationValue, RGBShift, ChannelShuffle,
    ToGray, RandomSizedCrop)


def weak_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(p=0.2, shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    ], p=p)


class Albu():
    def __call__(self, image):
        augmentation = aug()

        data = {"image": image}
        augmented = augmentation(**data)

        return augmented["image"]


class TestNormalize:
    def __call__(self, data):
        smooth = 1e-6

        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                data[b][c] = (data[b][c] - np.min(data[b][c]) + smooth) / (np.max(data[b][c]) - np.min(data[b][c]) + smooth)

        data = data * 2 - 1

        return data


if __name__=="__main__":
    img = cv2.imread('mj.png', 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (380, 256))

    for i in range(100):
        data = {"image": img}
        aug = weak_aug()
        augmented = aug(**data)

        out_img = augmented["image"]
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        cv2.imshow('img', out_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
