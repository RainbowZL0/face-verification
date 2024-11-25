import os
from copy import deepcopy

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from PIL import Image
from torchvision.transforms import v2
from torchvision.io import read_image

import torch
from pathlib import Path

IMG_FOLDER = Path("../a001_img")
P1 = IMG_FOLDER / Path("person0_0.jpg")
P2 = IMG_FOLDER / Path("person0_2.jpg")

# # If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN()


def test_mtcnn_data_flow(img_path_0: Path):
    # img = read_image(str(img_path_0))  # 过时的方法，只不过现在用的是旧版torchvision。新版建议用decode_image()读取图片
    img = Image.open(img_path_0)

    basename_two_parts = os.path.basename(img_path_0).split(".")
    file_name, file_extension = basename_two_parts[0], basename_two_parts[1]
    det_save_to_path = (IMG_FOLDER /
                        Path(file_name + "_det" + "." + file_extension))

    det_save_to_path = str(det_save_to_path)

    pil_image_list = [img, deepcopy(img)]

    # mtcnn做batch prediction，要求输入为list of PIL.Image
    # 一旦指定了 save_path (我传入的是det_save_to_path)，那么必须传入与list of PIL.Image长度相同的path列表
    # 否则返回的人脸detection list的长度不对。
    img_detection = mtcnn(pil_image_list)

    test_tensor = torch.stack(img_detection)

    # InceptionResnetV1 支持批量处理，(B, C, H, W)
    img_emb = resnet(test_tensor)
    return img_emb


def get_cos_similarity(emb1, emb2):
    emb1 /= torch.linalg.norm(emb1)
    emb2 /= torch.linalg.norm(emb2)
    return torch.matmul(emb1, emb2).item()


def get_dot_product_similarity(emb1, emb2):
    return torch.matmul(emb1, emb2).item()


def test_mtcnn_detect_method():
    print(help(mtcnn.detect))


def test_01():
    emb_list = list()
    for img_path in (P1, P2):
        emb_list.append(
            test_mtcnn_data_flow(img_path)
        )


if __name__ == '__main__':
    pass
    test_01()
    # test_mtcnn_detect_method()
