"""Module for handling video stream and sending responses to server"""

import threading

import cv2
import grpc

from modules.models.utils import transform_mask2rle
from modules.api.grpc_gen import health_pb2_grpc, health_pb2
from modules.pose_estimator.video_pose_estimation import VideoPoseEstimator


class Handler:
    def __init__(self, shape_predictor_path: str, open_port: str, grpc_url: str):
        self.video_pose_estimator = VideoPoseEstimator(
            shape_predictor_path=shape_predictor_path,
            open_port=open_port
        )

        stub = self.__create_stud(grpc_url=grpc_url)

        self.__thread_blink = self.__create_thread(
            function_to_execute=self.__send_blink
        )
        # self.__thread_mask_landmarks = self.__create_thread(
        #     function_to_execute=self.__send_mask_and_keypoints
        # )
        # self.__send_blink()

    #     self.__all_threads = [
    #         self.__thread_blink,
    #         self.__thread_mask_landmarks
    #     ]
    #
    # def start(self):
    #     for curr_thread in self.__all_threads:
    #         curr_thread.start()

    @staticmethod
    def __create_thread(function_to_execute):
        thread_video_stream = threading.Thread(target=function_to_execute)
        thread_video_stream.daemon = True
        thread_video_stream.start()

        return thread_video_stream

    @staticmethod
    def __create_stud(grpc_url):
        channel = grpc.insecure_channel(target=grpc_url)
        stub = health_pb2_grpc.HealthStub(channel=channel)

        return stub

    def __send_blink(self):
        while True:
            is_blinked = self.video_pose_estimator.get_eye_blink()

            if is_blinked:
                print(f'User blinked!')

    def __send_mask_and_keypoints(self):
        while True:
            human_mask = self.video_pose_estimator.get_human_segmentation()
            facial_landmarks = self.video_pose_estimator.get_facial_landmarks()

            rle_human_mask = transform_mask2rle(image=human_mask)

            print(rle_human_mask)


