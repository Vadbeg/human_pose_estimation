import time

from modules.pose_estimator.video_pose_estimation import VideoPoseEstimator

import cv2
import grpc

from scripts.face_alignment_testing import draw_facial_info
from scripts.human_segmentation import (
    transform_image, create_model,
    get_prediction_mask, mask_to_original_shape,
    draw_mask_on_image
)
from modules.api.grpc_gen import health_pb2_grpc, health_pb2
from modules.models.utils import transform_mask2rle, transform_rle2mask


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

    time.sleep(1)

    channel = grpc.insecure_channel(target='localhost:9999')
    stud = health_pb2_grpc.HealthStub(channel=channel)

    while True:  # show streamed images until Ctrl-C

        image = video_pose_estimator.video_processor.receive_frame()
        human_mask = video_pose_estimator.get_human_segmentation()
        # facial_landmarks = video_pose_estimator.get_facial_landmarks()
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
        rle_mask, original_shape = transform_mask2rle(image=human_mask)
        human_mask = transform_rle2mask(rle_mask=rle_mask, original_shape=original_shape)

        image = draw_mask_on_image(image=image, mask=human_mask)

        cv2.imshow('Image', image)  # 1 window for each RPi
        cv2.waitKey(1)
