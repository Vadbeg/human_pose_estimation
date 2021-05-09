"""Module for human segmentation"""

from typing import Tuple, Callable

import torch
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from people_segmentation.pre_trained_models import create_model


class HumanSegmentation:
    def __init__(self, model_name: str = 'Unet_2020-07-20'):
        self.__model = create_model(model_name=model_name)

        self.__image_transforms = self.__get_image_transforms()

    def __get_prediction_mask(self, image_tensor: torch.Tensor) -> np.ndarray:
        image_tensor = image_tensor.unsqueeze(0)

        self.__model.eval()
        with torch.no_grad():
            prediction = self.__model(image_tensor)
            prediction = prediction[0][0]

        prediction = torch.sigmoid(prediction)
        prediction = prediction > 0.7

        prediction = prediction.cpu().numpy().astype(np.uint8)

        return prediction

    def __transform_image(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = self.__image_transforms(image=image)['image']

        return image_tensor

    @staticmethod
    def __get_image_transforms() -> Callable:
        transforms = albu.Compose([
            albu.Normalize(p=1),
            albu.PadIfNeeded(min_height=None, min_width=None,
                             pad_height_divisor=32, pad_width_divisor=32,
                             p=1.0),
            ToTensorV2(p=1.0)
        ], p=1)

        return transforms

    @staticmethod
    def __mask_to_original_shape(mask: np.ndarray, original_shape: Tuple) -> np.ndarray:
        height = original_shape[0]
        width = original_shape[1]

        mask = mask[-height:, -width:]

        return mask

    def extract_segmentation(self, image: np.ndarray):
        image_tensor = self.__transform_image(image=image)

        res_mask = self.__get_prediction_mask(image_tensor=image_tensor)
        res_mask = self.__mask_to_original_shape(mask=res_mask, original_shape=image.shape)

        return res_mask
