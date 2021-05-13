"""Module for checking if head position is good"""

import math
from typing import Tuple, List

import numpy as np

from modules.models.utils import get_annotated_facial_landmarks


class HeadPositionChecker:
    def __init__(self, edge_value: float = 15.0):
        self.edge_value = edge_value

        self.max_value = 90

    @staticmethod
    def __get_nose_landmarks(facial_landmarks: np.ndarray) -> List[Tuple[int, int]]:
        annotated_facial_landmarks = get_annotated_facial_landmarks(
            landmarks=facial_landmarks
        )
        nose_landmarks = annotated_facial_landmarks['nose']

        return nose_landmarks

    def is_head_position_good(self, facial_landmarks):
        nose_landmarks = self.__get_nose_landmarks(
            facial_landmarks=facial_landmarks
        )

        top_nose_point, bottom_nose_point = self.__get_nose_line_points(
            nose_landmarks=nose_landmarks
        )

        angle = self.__get_angle_between_vertical_and_nose(
            top_nose_point=top_nose_point,
            bottom_nose_point=bottom_nose_point
        )

        if np.abs(self.max_value - angle) > self.edge_value:
            return False

        return True

    @staticmethod
    def __get_nose_line_points(
            nose_landmarks: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:

        top_nose_point = nose_landmarks[0]
        all_bottom_nose_points = nose_landmarks[3:9]

        bottom_nose_point = np.sum(np.array(all_bottom_nose_points), axis=0) / len(all_bottom_nose_points)
        bottom_nose_point = np.uint16(bottom_nose_point)

        return top_nose_point, bottom_nose_point

    @staticmethod
    def __get_angle_between_vertical_and_nose(
            top_nose_point: Tuple[int, int],
            bottom_nose_point: Tuple[int, int]
    ) -> float:
        tg_angle = np.abs(top_nose_point[1] - bottom_nose_point[1]) / \
                   np.abs(top_nose_point[0] - bottom_nose_point[0])

        angle = (math.atan(tg_angle) / np.pi) * 180

        return angle


