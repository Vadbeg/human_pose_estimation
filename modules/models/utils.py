"""Module with utilities for model and usage of model results"""

from typing import Dict, Tuple, List

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


def get_annotated_facial_landmarks(landmarks: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
    landmarks = landmarks[0]
    landmarks = [tuple(np.int16(curr_landmark)) for curr_landmark in landmarks]

    face_contour = list(landmarks[:17])
    left_eyebrow = list(landmarks[17:22])
    right_eyebrow = list(landmarks[22:27])
    nose = list(landmarks[27:36])
    left_eye = list(landmarks[36:42])
    right_eye = list(landmarks[42:48])
    lips = list(landmarks[48:])

    annotated_landmarks = {
        'face_contour': face_contour,
        'left_eyebrow': left_eyebrow,
        'right_eyebrow': right_eyebrow,
        'nose': nose,
        'left_eye': left_eye,
        'right_eye': right_eye,
        'lips': lips,
    }

    return annotated_landmarks

