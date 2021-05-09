"""Module for extracting info for pose estimation"""

import numpy as np

from modules.models.human_segmentation import HumanSegmentation
from modules.models.facial_landmarks import FacialLandmarksExtractor

from modules.receiver.video_processor import VideoProcessor


class VideoPoseEstimator:
    def __init__(self, open_port: str = '5555', ):
        self.video_processor = VideoProcessor(open_port=open_port)

        self.human_segmentation_model = HumanSegmentation()
        self.facial_landmarks_extractor = FacialLandmarksExtractor()

    def __get_frame(self) -> np.ndarray:
        frame = self.video_processor.receive_frame()

        if not isinstance(frame, np.ndarray) and frame is None:
            raise ValueError(f'No frames')

        return frame

    def get_human_segmentation(self):
        frame = self.__get_frame()

        segmentation_mask = self.human_segmentation_model.extract_segmentation(image=frame)

        return segmentation_mask

    def get_facial_landmarks(self):
        frame = self.__get_frame()

        facial_landmarks = self.facial_landmarks_extractor.extract_landmarks(image=frame)

        return facial_landmarks
