import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

import tensorflow as tf


IMG_FOLDER = Path("../../a001_img")
P1 = IMG_FOLDER / Path("person0_0.jpg")
P2 = IMG_FOLDER / Path("person0_2.jpg")


def start_verify():
    result = DeepFace.verify(
        img1_path=str(P1),
        img2_path=str(P2),
        model_name="Facenet512",
    )
    return result


def start_detection_and_alignment():
    result_1: list = DeepFace.extract_faces(
        img_path=str(P1),
        detector_backend="mtcnn",
    )

    result_2: list = DeepFace.extract_faces(
        img_path=str(P2),
        detector_backend="mtcnn",
    )

    if len(result_1) > 0:
        for face in result_1:
            (x, y, w, h) = (face.get("facial_area").get(key) for key in ["x", "y", "w", "h"])
            img = cv2.imread(str(P1))

            cv2.rectangle(
                img=img,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=(0, 255, 0),  # BGR
                thickness=2,
            )
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

    pass


def test_resize_figure():
    figure = plt.figure(dpi=300)
    figure.set_size_inches(10, 10)
    axes = figure.add_axes((0, 0, 1, 1))

    print(axes.bbox.bounds)
    print(axes.get_position())

    figure.set_size_inches(5, 5)
    print(axes.bbox.bounds)
    print(axes.get_position())


def tensorflow_cuda():
    # 检查是否识别到GPU
    if tf.config.list_physical_devices('GPU'):
        print("CUDA GPU 已识别!")
        print("GPU设备列表:", tf.config.list_physical_devices('GPU'))
    else:
        print("未识别到CUDA GPU")


if __name__ == '__main__':
    pass
    # rich_pprint(start_verify())
    # start_detection_and_alignment()
    # test_resize_figure()
    # tensorflow_cuda()
    
