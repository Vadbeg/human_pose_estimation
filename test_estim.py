import time

from modules.pose_estimator.video_pose_estimation import VideoPoseEstimator

import cv2
import grpc
import numpy as np

from scripts.face_alignment_testing import draw_facial_info
from scripts.human_segmentation import (
    transform_image, create_model,
    get_prediction_mask, mask_to_original_shape,
    draw_mask_on_image
)
from modules.api.grpc_gen import health_pb2_grpc, health_pb2
from modules.models.utils import transform_mask2rle, transform_rle2mask, get_annotated_facial_landmarks
from modules.pose_estimator.shoulders_position import ShouldersPositionChecker
from modules.pose_estimator.head_position import HeadPositionChecker


def make_message():
    return health_pb2.Blinked(amount=1)


def generate_messages():
    messages = [
        make_message(),
    ]

    for idx, curr_msg in enumerate(messages):
        print(f'{idx}')

        yield curr_msg


if __name__ == '__main__':
    video_pose_estimator = VideoPoseEstimator(
        shape_predictor_path='files/shape_predictor_68_face_landmarks.dat',
        open_port='5554'
    )

    shoulders_position_checker = ShouldersPositionChecker()
    head_position_checker = HeadPositionChecker()

    time.sleep(1)

    channel = grpc.insecure_channel(target='localhost:9999')
    stud = health_pb2_grpc.HealthStub(channel=channel)

    asked = False

    while True:  # show streamed images until Ctrl-C

        image = video_pose_estimator.video_processor.receive_frame()
        # human_mask = video_pose_estimator.get_human_segmentation()
        facial_landmarks = video_pose_estimator.get_facial_landmarks()
        # is_blink = video_pose_estimator.get_eye_blink()

        # image = draw_facial_info(image=image, facial_info=facial_landmarks)
        #
        # print(is_blink)
        #
        # if is_blink:
        #     # blinked = health_pb2.Blinked()
        #
        #     feature = stud.UserBlinked(generate_messages())
        #
        #     print(feature)
        #
        #     image = cv2.putText(
        #         image, 'Blink detected', (10, 30),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        #     )

        # is_good_shoulders_pos = shoulders_position_checker.is_shoulders_position_good(
        #     human_mask=human_mask,
        #     facial_landmarks=facial_landmarks
        # )
        #
        # image = cv2.putText(
        #     image, f'Is good position: {is_good_shoulders_pos}', (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        # )

        is_good_head_pos = head_position_checker.is_head_position_good(facial_landmarks=facial_landmarks)

        image = cv2.putText(
            image, f'Is good head pos: {is_good_head_pos}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        # image = draw_facial_info(image=image, facial_info=facial_landmarks)

        # image = draw_mask_on_image(image=image, mask=human_mask)

        cv2.imshow('Image', image)  # 1 window for each RPi
        cv2.waitKey(1)

