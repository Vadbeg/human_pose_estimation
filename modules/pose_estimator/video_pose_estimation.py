"""Module for extracting info for pose estimation"""

from typing import Optional

import numpy as np

from modules.models.human_segmentation import HumanSegmentation
from modules.models.facial_landmarks import FacialLandmarksExtractor
from modules.models.blink_detector import BlinkDetector

from modules.receiver.video_processor import VideoProcessor

from modules.pose_estimator.head_position import HeadPositionChecker
from modules.pose_estimator.shoulders_position import ShouldersPositionChecker


class VideoPoseEstimator:
    def __init__(self, shape_predictor_path: str, open_port: str = '5555'):
        self.video_processor = VideoProcessor(open_port=open_port)

        self.blink_detector = BlinkDetector(shape_predictor_path=shape_predictor_path)
        self.human_segmentation_model = HumanSegmentation()
        self.facial_landmarks_extractor = FacialLandmarksExtractor()

        self.head_position_checker = HeadPositionChecker(edge_value=15.0)
        self.shoulders_position_checker = ShouldersPositionChecker(ratio=1.5)

    def __get_frame(self) -> np.ndarray:
        frame = self.video_processor.receive_frame()

        if not isinstance(frame, np.ndarray) and frame is None:
            raise ValueError(f'No frames')

        return frame

    def __get_human_segmentation(self, frame: np.ndarray):
        segmentation_mask = self.human_segmentation_model.extract_segmentation(image=frame)

        return segmentation_mask

    def __get_facial_landmarks(self, frame: np.ndarray):
        facial_landmarks = self.facial_landmarks_extractor.extract_landmarks(image=frame)

        return facial_landmarks

    def check_eye_blink(self) -> Optional[bool]:
        frame = self.__get_frame()

        is_blink = self.blink_detector.detect_blink(image=frame)

        return is_blink

    def check_shoulders_position(self) -> Optional[bool]:
        frame = self.__get_frame()

        human_mask = self.__get_human_segmentation(frame=frame)
        facial_landmarks = self.__get_facial_landmarks(frame=frame)

        if facial_landmarks is None:
            return True

        is_good_shoulder_position = self.shoulders_position_checker.is_shoulders_position_good(
            human_mask=human_mask,
            facial_landmarks=facial_landmarks
        )

        return is_good_shoulder_position

    def check_head_position(self) -> Optional[bool]:
        frame = self.__get_frame()

        facial_landmarks = self.__get_facial_landmarks(frame=frame)

        if facial_landmarks is None:
            return None

        is_good_head_position = self.head_position_checker.is_head_position_good(
            facial_landmarks=facial_landmarks
        )

        return is_good_head_position
