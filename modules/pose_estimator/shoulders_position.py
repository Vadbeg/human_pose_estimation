"""Module for checking if shoulders position is good"""

from typing import Tuple

import cv2
import numpy as np

from modules.models.utils import get_annotated_facial_landmarks


class ShouldersPositionChecker:
    def __init__(self, ratio: float = 1.5):
        self.ratio = ratio

    def is_shoulders_position_good(self, human_mask: np.ndarray, facial_landmarks: np.ndarray) -> bool:
        min_face_contour_landmark = self.__get_min_face_contour_landmark(
            facial_landmarks=facial_landmarks
        )

        human_mask_left, human_mask_right = self.__get_image_parts(
            human_mask=human_mask,
            min_point=min_face_contour_landmark
        )

        left_contour_area = self.__calculate_max_contour_area(mask=human_mask_left)
        right_contour_area = self.__calculate_max_contour_area(mask=human_mask_right)

        if (left_contour_area / right_contour_area > 1.5) \
                or (right_contour_area / left_contour_area > 1.5):
            return False

        return True

    @staticmethod
    def __get_min_face_contour_landmark(facial_landmarks: np.ndarray) -> Tuple[int, int]:
        annotated_facial_landmarks = get_annotated_facial_landmarks(
            landmarks=facial_landmarks
        )
        face_contour_landmarks = annotated_facial_landmarks['face_contour']

        min_face_contour_landmark = max(face_contour_landmarks, key=lambda x: x[1])

        return min_face_contour_landmark

    @staticmethod
    def __get_image_parts(
            human_mask: np.ndarray,
            min_point: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        human_mask_abdomen = human_mask[min_point[1]:, :]

        human_mask_left = human_mask_abdomen[:, :min_point[0]]
        human_mask_right = human_mask_abdomen[:, min_point[0]:]

        return human_mask_left, human_mask_right

    @staticmethod
    def __calculate_max_contour_area(mask: np.ndarray) -> int:
        contours, _ = cv2.findContours(
            image=mask,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_NONE
        )

        contours_area = [cv2.contourArea(curr_contour) for curr_contour in contours]
        max_contour_are = max(contours_area)

        return max_contour_are
