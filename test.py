from modules.api.handler import Handler


if __name__ == '__main__':
    handler = Handler(
        open_port='5554',
        shape_predictor_path='files/shape_predictor_68_face_landmarks.dat',
        grpc_url='localhost:9999'
    )

    # handler.start()
