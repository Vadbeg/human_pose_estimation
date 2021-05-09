"""Module for processing video received via tcp"""

import threading

import cv2
import imagezmq


class VideoProcessor:
    def __init__(self, open_port='5555'):
        self.__frame = None
        self.__stop = False

        self.__open_port = open_port

        self.__frame_ready = threading.Event()
        self.__thread_video_stream = self.__create_thread(
            function_to_execute=self.__handle_video_stream
        )

    @staticmethod
    def __create_thread(function_to_execute):
        thread_video_stream = threading.Thread(target=function_to_execute)
        thread_video_stream.daemon = True
        thread_video_stream.start()

        return thread_video_stream

    def receive_frame(self, timeout=30.0):
        flag = self.__frame_ready.wait(timeout=timeout)

        if not flag:
            raise TimeoutError(f'Timeout while reading from subscriber: tcp://*:{self.__open_port}')

        self.__frame_ready.clear()

        return self.__frame

    def __handle_video_stream(self):
        image_hub = imagezmq.ImageHub(open_port=f'tcp://localhost:{self.__open_port}', REQ_REP=False)

        while not self.__stop:
            frame_name, frame = image_hub.recv_image()

            self.__frame = frame
            self.__frame_ready.set()

        image_hub.close()

    def close(self):
        self.__stop = True


