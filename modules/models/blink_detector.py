"""Model for blink detection"""


import cv2
import dlib
import imutils
import imutils.face_utils
import numpy as np
import scipy.spatial


class BlinkDetector:
    def __init__(
            self, shape_predictor_path: str,
            threshold: float = 0.23,
            consecutive_frames_num: int = 5,
    ):
        self.__detector = dlib.get_frontal_face_detector()
        self.__predictor = dlib.shape_predictor(shape_predictor_path)

        self.__threshold = threshold
        self.__consecutive_frames_num = consecutive_frames_num

    @staticmethod
    def __eye_aspect_ratio(eye):
        point_a = scipy.spatial.distance.euclidean(eye[1], eye[5])
        point_b = scipy.spatial.distance.euclidean(eye[2], eye[4])

        point_c = scipy.spatial.distance.euclidean(eye[0], eye[3])

        ear = (point_a + point_b) / (2.0 * point_c)

        return ear

    def detect_blink(self, image: np.ndarray) -> bool:
        image = imutils.resize(image, width=450)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = self.__detector(gray, 0)

        (l_start, l_end) = imutils.face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (r_start, r_end) = imutils.face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        for curr_rect in rects:
            ear = self.__check_eyes_aspect_ratio(
                image_gray=gray,
                rectangle=curr_rect,
                l_start=l_start, l_end=l_end,
                r_start=r_start, r_end=r_end
            )

            if ear < self.__threshold:
                return True

        return False

    def __check_eyes_aspect_ratio(
            self, image_gray, rectangle,
            l_start, l_end,
            r_start, r_end
    ) -> float:
        shape = self.__predictor(image_gray, rectangle)
        shape = imutils.face_utils.shape_to_np(shape)

        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        left_ear = self.__eye_aspect_ratio(left_eye)
        right_ear = self.__eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        return ear


