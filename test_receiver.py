import imagezmq


if __name__ == '__main__':
    image_hub = imagezmq.ImageHub(open_port='tcp://*:5555')

    while True:  # show streamed images until Ctrl-C
        rpi_name, image = image_hub.recv_image()

        image_hub.send_reply(b'OK')
