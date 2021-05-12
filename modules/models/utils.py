"""Module with utilities for model and usage of model results"""

from typing import List, Tuple

import numpy as np


def transform_mask2rle(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    pixels = image.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])
    runs = runs[0] + 1

    runs[1::2] -= runs[::2]

    original_shape = image.shape

    return runs, original_shape


def transform_rle2mask(rle_mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:

    starts, lengths = [
        np.asarray(x, dtype=int)
        for x in (rle_mask[0:][::2], rle_mask[1:][::2])
    ]

    starts -= 1
    ends = starts + lengths

    image = np.zeros(shape=original_shape[0] * original_shape[1], dtype=np.uint8)

    for curr_start, curr_end in zip(starts, ends):
        image[curr_start: curr_end] = 1

    image = image.reshape(original_shape, order='F')

    return image



