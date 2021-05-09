import os
from typing import List, Tuple, Dict

import cv2
import imagezmq
import face_alignment

from scripts.face_alignment_testing import draw_facial_info
from scripts.human_segmentation import (
    transform_image, create_model,
    get_prediction_mask, mask_to_original_shape,
    draw_mask_on_image
)


if __name__ == '__main__':
    image_hub = imagezmq.ImageHub()

    while True:  # show streamed images until Ctrl-C
        rpi_name, image = image_hub.recv_image()

        print(image)

        image_hub.send_reply(b'OK')
