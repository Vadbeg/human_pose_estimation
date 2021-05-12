"""Script for face detetion and facial landmarks retrieving"""

import argparse
from typing import List, Tuple, Dict

import torch
import numpy as np
import face_alignment
from cv2 import cv2
from tqdm import tqdm


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(filename=image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def draw_bbox(image: np.ndarray, bbox_coords: List[int]):
    x1 = bbox_coords[0]
    y1 = bbox_coords[1]

    width = bbox_coords[2]
    height = bbox_coords[3]

    x2 = x1 + width
    y2 = y1 + height

    pt1 = (x1, y1)
    pt2 = (x2, y2)

    image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), 3)

    return image


def draw_landmarks(image: np.ndarray, landmarks: Dict[str, Tuple[int, int]]):

    for curr_landmark in landmarks:
        curr_landmark = tuple([int(curr_coord) for curr_coord in curr_landmark])

        image = cv2.circle(image, curr_landmark, 3, (255, 0, 0), -1)

    return image


def draw_facial_info(image: np.ndarray, facial_info: List[Dict]):

    for curr_face_info in facial_info:
        # bbox_coords = curr_face_info['box']
        # landmarks = curr_face_info['keypoints']

        # image = draw_bbox(image=image, bbox_coords=bbox_coords)
        image = draw_landmarks(image=image, landmarks=curr_face_info)

    return image


def parse_arg():
    parser = argparse.ArgumentParser(description=f'Script for face detection and landmark retrieving')

    parser.add_argument('--image-path', type=str, help=f'Path to image file')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arg()

    fa_model = face_alignment.FaceAlignment(
        landmarks_type=face_alignment.LandmarksType._2D,
    )

    image_path = args.image_path
    image = read_image(image_path=image_path)
    # image = np.transpose(image, axes=(2, 1, 0))

    image_batch = np.concatenate([image[np.newaxis, :]] * 10, axis=0)
    image_batch = torch.tensor(image_batch)

    print(image_batch.size())

    res_facial_info = fa_model.get_landmarks(image_or_path=image)

    # res_facial_info = [res_facial_info[0][:17]]
    # res_facial_info = [res_facial_info[0][17:22]]
    # res_facial_info = [res_facial_info[0][22:27]]
    # res_facial_info = [res_facial_info[0][36:42]]
    # res_facial_info = [res_facial_info[0][42:48]]
    res_facial_info = [res_facial_info[0][48:]]

    print(res_facial_info)

    res_image = draw_facial_info(image=image, facial_info=res_facial_info)
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    cv2.imshow('Face', res_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


