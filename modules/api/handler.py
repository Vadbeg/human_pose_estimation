"""Module for handling video stream and sending responses to server"""

import time
import threading

import grpc

from modules.api.grpc_gen import health_pb2_grpc, health_pb2
from modules.pose_estimator.video_pose_estimation import VideoPoseEstimator


class Handler:
    def __init__(self, shape_predictor_path: str, open_port: str, grpc_url: str):
        self.video_pose_estimator = VideoPoseEstimator(
            shape_predictor_path=shape_predictor_path,
            open_port=open_port
        )

        self.stub = self.__create_stub(grpc_url=grpc_url)

        self.__thread_blink = self.__create_thread(
            function_to_execute=self.__send_blink
        )
        self.__thread_shoulders = self.__create_thread(
            function_to_execute=self.__send_shoulders_position
        )
        self.__thread_head = self.__create_thread(
            function_to_execute=self.__send_head_position
        )

        self.__all_threads = [
            self.__thread_blink,
            self.__thread_shoulders,
            self.__thread_head
        ]

    def start(self):
        for curr_thread in self.__all_threads:
            curr_thread.start()

        for curr_thread in self.__all_threads:
            curr_thread.join()

    @staticmethod
    def __create_thread(function_to_execute):
        thread_video_stream = threading.Thread(target=function_to_execute)
        thread_video_stream.daemon = True

        return thread_video_stream

    @staticmethod
    def __create_stub(grpc_url):
        channel = grpc.insecure_channel(target=grpc_url)
        stub = health_pb2_grpc.HealthStub(channel=channel)

        return stub

    def __create_blink_msg(self):
        while True:
            time.sleep(0.05)

            is_blink = self.video_pose_estimator.check_eye_blink()
            is_face_recognized = not (is_blink is None)

            if is_blink or not is_face_recognized:
                blinked_msg = health_pb2.Blinked(
                    amount=1,
                    isFaceRecognized=is_face_recognized
                )

                yield blinked_msg

    def __create_shoulders_position_msg(self):
        while True:
            try:

                is_good_shoulder_position = self.video_pose_estimator.check_shoulders_position()
                is_face_recognized = not (is_good_shoulder_position is None)

                if is_good_shoulder_position is None:
                    is_good_shoulder_position = False

                shoulder_pos_change_msg = health_pb2.ShouldersPositionChangeMsg(
                    isCrooked=not is_good_shoulder_position,
                    isFaceRecognized=is_face_recognized
                )

                yield shoulder_pos_change_msg
            except Exception as exc:
                print(exc)

    def __create_head_position_msg(self):
        while True:
            time.sleep(0.5)

            is_good_head_position = self.video_pose_estimator.check_head_position()
            is_face_recognized = not (is_good_head_position is None)

            if is_good_head_position is None:
                is_good_head_position = False

            nose_pos_change_msg = health_pb2.NosePositionChangeMsg(
                isCrooked=not is_good_head_position,
                isFaceRecognized=is_face_recognized
            )

            yield nose_pos_change_msg

    def __send_blink(self):
        self.stub.UserBlinked(self.__create_blink_msg())

    def __send_shoulders_position(self):
        self.stub.ShouldersPositionChange(self.__create_shoulders_position_msg())

    def __send_head_position(self):
        self.stub.NosePositionChange(self.__create_head_position_msg())






