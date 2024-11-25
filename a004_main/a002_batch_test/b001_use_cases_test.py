import random

from rich.pretty import pprint
from torch.utils.data import DataLoader

from a004_main.a001_utils.a000_CONFIG import (
    DATASET_SF_TL54_PATH,
    TEST_BATCH_SIZE,
)
from a004_main.a002_batch_test.a002_DatasetForTestOrVali import DatasetForTestOrVali


def test_my_dataset():
    """测试用例"""
    my_dataset = DatasetForTestOrVali(
        dataset_path=DATASET_SF_TL54_PATH,
        num_samples_per_epoch=1000,
    )
    for i in range(20):
        rand_index = random.randint(0, 999)
        pprint(my_dataset[rand_index])


def test_dataloader():
    """测试用例"""
    my_dataset = DatasetForTestOrVali(
        dataset_path=DATASET_SF_TL54_PATH,
        num_samples_per_epoch=1000,
    )
    dtl = DataLoader(
        dataset=my_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    for batch in dtl:
        pprint(batch)
        break


def test_deep_face():
    from deepface import DeepFace
    import cv2
    img = cv2.imread('..\\..\\a000_DATASET\\SF-TL54\\gray\\train\\images\\1_1_2_1_1097_116_1.png')
    face_dict_list = DeepFace.extract_faces(
        img_path=img,
        detector_backend='mtcnn',
        enforce_detection=False,
    )
    pass


if __name__ == '__main__':
    test_deep_face()
    pass
