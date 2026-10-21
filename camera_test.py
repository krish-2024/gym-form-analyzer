import logging

import cv2

LOGGER = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    backends = (
        ("DirectShow", cv2.CAP_DSHOW),
        ("Microsoft Media Foundation", cv2.CAP_MSMF),
    )

    for index in range(3):
        for backend_name, backend in backends:
            camera = None
            try:
                camera = cv2.VideoCapture(index, backend)
                opened = camera.isOpened()
                if opened:
                    frame_read, _ = camera.read()
                else:
                    frame_read = False

                LOGGER.info(
                    f"Camera index {index} ({backend_name}): "
                    f"opened={'yes' if opened else 'no'}, "
                    f"frame read={'yes' if frame_read else 'no'}"
                )
            except cv2.error as error:
                LOGGER.error(
                    "Camera index %s (%s): OpenCV error: %s",
                    index,
                    backend_name,
                    error,
                )
            finally:
                if camera is not None:
                    camera.release()


if __name__ == "__main__":
    main()
