"""Script for human segmentation"""

import argparse
from typing import Tuple

import torch
import numpy as np
import albumentations as albu
from cv2 import cv2
from people_segmentation.pre_trained_models import create_model
from albumentations.pytorch.transforms import ToTensorV2


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(filename=image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def transform_image(image: np.ndarray) -> torch.Tensor:
    transform = albu.Compose([
        albu.Normalize(p=1),
        albu.PadIfNeeded(min_height=None, min_width=None,
                         pad_height_divisor=32, pad_width_divisor=32,
                         p=1.0),
        ToTensorV2(p=1.0)
    ], p=1)

    image_tensor = transform(image=image)['image']

    return image_tensor


def mask_to_original_shape(mask: np.ndarray, original_shape: Tuple) -> np.ndarray:
    height = original_shape[0]
    width = original_shape[1]

    mask = mask[-height:, -width:]

    return mask


def get_prediction_mask(image_tensor: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = prediction[0][0]


    prediction = torch.sigmoid(prediction)

    print(prediction.max())
    print(prediction.min())

    prediction = (prediction > 0.7).cpu().numpy().astype(np.uint8)

    return prediction


def parse_arg():
    parser = argparse.ArgumentParser(description=f'Script for face detection and landmark retrieving')

    parser.add_argument('--image-path', type=str, help=f'Path to image file')

    args = parser.parse_args()

    return args


def draw_mask_on_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8)

    image_mask = cv2.addWeighted(image, 1, mask, 0.5, 0)
    image_mask = np.uint8(image_mask)

    return image_mask


if __name__ == '__main__':
    args = parse_arg()

    image_path = args.image_path
    image = read_image(image_path=image_path)

    image_tensor = transform_image(image=image)

    model = create_model(model_name='Unet_2020-07-20')

    res_mask = get_prediction_mask(image_tensor=image_tensor, model=model)
    res_mask = mask_to_original_shape(mask=res_mask, original_shape=image.shape)

    image_with_mask = draw_mask_on_image(image=image, mask=res_mask)

    image_with_mask = cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB)

    cv2.imshow('Mask', image_with_mask)
    cv2.waitKey(0)


