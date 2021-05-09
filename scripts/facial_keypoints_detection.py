"""Script for face detetion and facial landmarks retrieving"""

import argparse
from typing import List, Tuple, Dict

import numpy as np
from cv2 import cv2
from mtcnn import MTCNN


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
    image = cv2.circle(image, (landmarks['left_eye']), 3, (255, 0, 0), -1)
    image = cv2.circle(image, (landmarks['right_eye']), 3, (255, 0, 0), -1)
    image = cv2.circle(image, (landmarks['nose']), 3, (255, 0, 0), -1)
    image = cv2.circle(image, (landmarks['mouth_left']), 3, (255, 0, 0), -1)
    image = cv2.circle(image, (landmarks['mouth_right']), 3, (255, 0, 0), -1)

    return image


def draw_facial_info(image: np.ndarray, facial_info: List[Dict]):

    for curr_face_info in facial_info:
        bbox_coords = curr_face_info['box']
        landmarks = curr_face_info['keypoints']

        image = draw_bbox(image=image, bbox_coords=bbox_coords)
        image = draw_landmarks(image=image, landmarks=landmarks)

    return image


def parse_arg():
    parser = argparse.ArgumentParser(description=f'Script for face detection and landmark retrieving')

    parser.add_argument('--image-path', type=str, help=f'Path to image file')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arg()

    detector = MTCNN()

    image_path = args.image_path
    image = read_image(image_path=image_path)

    res_facial_info = detector.detect_faces(img=image)

    res_image = draw_facial_info(image=image, facial_info=res_facial_info)

    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    cv2.imshow('Face', res_image)
    cv2.waitKey(0)


