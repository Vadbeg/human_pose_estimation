"""Module for extracting facial landmarks from image"""

import numpy as np
import face_alignment


class FacialLandmarksExtractor:
    def __init__(self):
        self.model = face_alignment.FaceAlignment(
            landmarks_type=face_alignment.LandmarksType._2D
        )

    def extract_landmarks(self, image: np.ndarray) -> np.ndarray:
        res_facial_info = self.model.get_landmarks(image_or_path=image)

        return res_facial_info

