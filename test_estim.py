import time

from modules.pose_estimator.video_pose_estimation import VideoPoseEstimator

import cv2

from scripts.face_alignment_testing import draw_facial_info
from scripts.human_segmentation import (
    transform_image, create_model,
    get_prediction_mask, mask_to_original_shape,
    draw_mask_on_image
)


if __name__ == '__main__':
    video_pose_estimator = VideoPoseEstimator(open_port='5554')

    human_mask = video_pose_estimator.get_human_segmentation()
    facial_landmarks = video_pose_estimator.get_facial_landmarks()

    time.sleep(1)

    while True:  # show streamed images until Ctrl-C

        image = video_pose_estimator.video_processor.receive_frame()
        # human_mask = video_pose_estimator.get_human_segmentation()
        facial_landmarks = video_pose_estimator.get_facial_landmarks()

        try:
            image = draw_facial_info(image=image, facial_info=facial_landmarks)
            # image = draw_mask_on_image(image=image, mask=human_mask)
        except Exception:
            pass

        cv2.imshow('Image', image)  # 1 window for each RPi
        cv2.waitKey(1)
