from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from colorama import Fore
from deepface import DeepFace
from facenet_pytorch import MTCNN
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm import tqdm

from copy import deepcopy

from a002_DatasetForTestOrVali import DatasetForTestOrVali
from a004_main.a001_utils.a000_CONFIG import *
from a004_main.a001_utils.a002_general_utils import (
    my_collate_fn_factory,
    save_to_json,
    build_dataset_for_test,
    init_a_figure_and_an_axes,
    adjust_figure_size_and_show_image_and_release_resources,
    load_json,
)


class MyTestObj:
    def __init__(self, dataset_for_test_obj: DatasetForTestOrVali):
        self.dataset_for_test_obj = dataset_for_test_obj
        self.dataloader = self.get_dataloader()
        self.result_recorder_list = list()

    def get_dataloader(self):
        return DataLoader(
            dataset=self.dataset_for_test_obj,
            batch_size=TEST_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            collate_fn=my_collate_fn_factory(TEST_BATCH_SIZE),
        )

    def test_detection(self):
        for index, batch_dict in tqdm(enumerate(self.dataloader)):
            batch_dict: dict
            test_detection_deepface_for_one_batch(
                batch_dict=batch_dict,
            )
            # test_detection_facenet_pytorch_for_one_batch(
            #     batch_dict=batch_dict,
            # )

    def test_verification_deepface(self):
        """
        result_dict = {
            'img0_path': '..\\..\\a000_DATASET\\SF-TL54\\gray\\test\\images\\119_1_2_8_144_102_1.png',
            'img1_path': '..\\..\\a000_DATASET\\SF-TL54\\gray\\test\\images\\119_1_2_1_115_32_1.png',
            'person0_key': 'person_119',
            'person1_key': 'person_119',
            'mod0': 'infrared',
            'mod1': 'infrared',
            'same_modality': True,
            'label': True,
            'model_prediction_result': True or False,
            'prediction_compared_with_label': True or False,
        }
        """
        for index, batch_dict in tqdm(enumerate(self.dataloader)):
            batch_dict: dict
            # batch_dict形式如下，
            # {
            # │   'img0_path': '..\\..\\a000_DATASET\\SF-TL54\\gray\\test\\images\\119_1_2_8_144_102_1.png',
            # │   'img1_path': '..\\..\\a000_DATASET\\SF-TL54\\gray\\test\\images\\119_1_2_1_115_32_1.png',
            # │   'person0_key': 'person_119',
            # │   'person1_key': 'person_119',
            # │   'mod0': 'infrared',
            # │   'mod1': 'infrared',
            # │   'same_modality': True,
            # │   'label': True
            # }

            # pprint(batch_dict)

            verify_dict = DeepFace.verify(
                img1_path=batch_dict["img0_path"],
                img2_path=batch_dict["img1_path"],
                model_name=TEST_MODEL_NAME,
                detector_backend=TEST_DETECTOR_NAME,
                enforce_detection=False,
            )
            model_prediction_result = verify_dict["verified"]
            # batch_dict["label"]: Tensor
            label = batch_dict["label"]
            prediction_compared_with_label = True if model_prediction_result is label else False

            # 往batch_dict添加两项，成为result_dict
            result_dict = deepcopy(batch_dict)
            result_dict["model_prediction_result"] = model_prediction_result
            result_dict["prediction_compared_with_label"] = prediction_compared_with_label
            self.result_recorder_list.append(result_dict)

            pprint(result_dict)

    def test_verification_facenet_pytorch(self):
        pass


def test_detection_deepface_for_one_batch(batch_dict: dict):
    """
    batch_dict形式如下，
    {
    │   'img0_path': '..\\..\\a000_DATASET\\SF-TL54\\gray\\test\\images\\119_1_2_8_144_102_1.png',
    │   'img1_path': '..\\..\\a000_DATASET\\SF-TL54\\gray\\test\\images\\119_1_2_1_115_32_1.png',
    │   'person0_key': 'person_119',
    │   'person1_key': 'person_119',
    │   'mod0': 'infrared',
    │   'mod1': 'infrared',
    │   'same_modality': True,
    │   'label': True
    }
    """
    batch_dict: dict
    # 注意cv2.imread()返回格式为 BGR uint8 HxWxC
    img0 = cv2.imread(batch_dict["img0_path"])
    img1 = cv2.imread(batch_dict["img1_path"])

    # extract_faces接受BGR uint8。
    # img0_face_list每个元素是一个字典，对应img0上的每个人脸
    try:
        img0_face_list = DeepFace.extract_faces(
            img_path=img0,
            detector_backend=TEST_DETECTOR_NAME,
            enforce_detection=False,
        )
        img1_face_list = DeepFace.extract_faces(
            img_path=img1,
            detector_backend=TEST_DETECTOR_NAME,
            enforce_detection=False,
        )
    except Exception as e:
        LOGGER.warn(
            Fore.LIGHTRED_EX +
            f"{e}"
        )
    else:
        packed_as_dict = {
            "0": {
                "img": img0,
                "face_list": img0_face_list,
            },
            "1": {
                "img": img1,
                "face_list": img1_face_list,
            }
        }

        figure, axes = init_a_figure_and_an_axes()

        for value_dict in packed_as_dict.values():
            # value_dict: dict {
            #   "img": ndarray,
            #   "face_list": list of face_dict
            # }
            for face in value_dict["face_list"]:
                # face: dict {
                #   "face": ndarray,  # 注意格式是 BGR float HxWxC
                #   "facial_area": {},
                #   "confidence": float,
                # }

                # 框中人脸显示
                x, y, w, h = (face.get("facial_area").get(key) for key in ["x", "y", "w", "h"])
                cv2.rectangle(
                    img=value_dict["img"],
                    pt1=(x, y),
                    pt2=(x + w, y + h),
                    color=(0, 255, 0),  # BGR
                    thickness=2,
                )
                adjust_figure_size_and_show_image_and_release_resources(
                    img=cv2.cvtColor(value_dict["img"], cv2.COLOR_BGR2RGB),
                    figure=figure,
                    axes=axes,
                )

                # 仅显示人脸
                face_array = face.get("face")  # float RGB
                # if face_array.dtype == np.float32 or face_array.dtype == np.float64:
                #     face_array = (face_array * 255).astype(np.uint8)  # 归一化到[0, 255]
                adjust_figure_size_and_show_image_and_release_resources(
                    img=face_array,
                    figure=figure,
                    axes=axes,
                )


def test_detection_facenet_pytorch_for_one_batch(batch_dict: dict):
    """
    batch_dict形式如下（当batch_size = 1），
    {
    │   'img0_path': '..\\..\\a000_DATASET\\SF-TL54\\gray\\test\\images\\119_1_2_8_144_102_1.png',
    │   'img1_path': '..\\..\\a000_DATASET\\SF-TL54\\gray\\test\\images\\119_1_2_1_115_32_1.png',
    │   'person0_key': 'person_119',
    │   'person1_key': 'person_119',
    │   'mod0': 'infrared',
    │   'mod1': 'infrared',
    │   'same_modality': True,
    │   'label': True
    }
    如果batch_size > 1, 预期将得到同样结构的字典，但每个value都变成list。如果这方面有问题，去检查my_collate_fn()

    mtcnn做batch prediction，要求输入为list of PIL.Image，一旦指定了 save_path (我传入的是
    det_save_to_path)，那么必须传入与list of PIL.Image长度相同的path列表，否则返回的人脸
    detection list的长度不对。

    mtcnn返回人脸结果，3x160x160，有负值（推测已经经过normalization）。
    然而deepface的detection方法extract_faces()返回的数值只有正数。
    """
    mtcnn = MTCNN(device=TEST_DEVICE)

    # 处理figure上的axes。如果没有则创建axes。
    figure, axes = init_a_figure_and_an_axes()

    if TEST_BATCH_SIZE == 1:
        img0, img1 = (Image.open(batch_dict[key]) for key in ["img0_path", "img1_path"])

        debug_pil_image_list = [img0, img1]
        face_list = mtcnn(img0)
        face: torch.Tensor

        # 如果face是None，说明没有检测到人脸
        for face in face_list:
            if face is None:
                # TODO
                return

        # 从CxHxW到HxWxC，其中C的顺序都是RGB
        face_list = [face.permute(dims=(1, 2, 0)).numpy() for face in face_list]

        # 显示face
        for face in face_list:
            face: np.ndarray
            adjust_figure_size_and_show_image_and_release_resources(
                img=face,
                figure=figure,
                axes=axes,
            )


def start_main_detection_and_alignment_test():
    my_dataset, my_verification_obj = build_dataset_and_test_obj()
    my_verification_obj.test_detection()


def analyze_result(lst):
    # infrared and vis
    diff_mod_tp = 0
    diff_mod_tn = 0
    diff_mod_fp = 0
    diff_mod_fn = 0

    # vis and vis
    vis_tp = 0
    vis_tn = 0
    vis_fp = 0
    vis_fn = 0

    # infra and infra
    infrared_tp = 0
    infrared_tn = 0
    infrared_fp = 0
    infrared_fn = 0

    for result_dict in lst:
        if result_dict["mod0"] == "infrared" and result_dict["mod1"] == "infrared":
            (infrared_tp,
             infrared_tn,
             infrared_fp,
             infrared_fn) = __confusion_matrix_branch(
                result_dict=result_dict,
                tp=infrared_tp,
                tn=infrared_tn,
                fp=infrared_fp,
                fn=infrared_fn,
            )
        if result_dict["mod0"] == "infrared" and result_dict["mod1"] == "vis":
            (diff_mod_tp,
             diff_mod_tn,
             diff_mod_fp,
             diff_mod_fn) = __confusion_matrix_branch(
                result_dict=result_dict,
                tp=diff_mod_tp,
                tn=diff_mod_tn,
                fp=diff_mod_fp,
                fn=diff_mod_fn,
            )
        if result_dict["mod0"] == "vis" and result_dict["mod1"] == "vis":
            vis_tp, vis_tn, vis_fp, vis_fn = __confusion_matrix_branch(
                result_dict=result_dict,
                tp=vis_tp,
                tn=vis_tn,
                fp=vis_fp,
                fn=vis_fn,
            )

    final_result_dict = {
        "diff-mod": {
            "tp": diff_mod_tp,
            "tn": diff_mod_tn,
            "fp": diff_mod_fp,
            "fn": diff_mod_fn,
        },
        "infrared": {
            "tp": infrared_tp,
            "tn": infrared_tn,
            "fp": infrared_fp,
            "fn": infrared_fn,
        },
        "vis": {
            "tp": vis_tp,
            "tn": vis_tn,
            "fp": vis_fp,
            "fn": vis_fn,
        },
    }

    pprint(final_result_dict)

    return final_result_dict


def __confusion_matrix_branch(result_dict, tp, tn, fp, fn):
    if result_dict["label"] and result_dict["model_prediction_result"]:
        tp += 1
    if not result_dict["label"] and not result_dict["model_prediction_result"]:
        tn += 1
    if not result_dict["label"] and result_dict["model_prediction_result"]:
        fp += 1
    if result_dict["label"] and not result_dict["model_prediction_result"]:
        fn += 1
    return tp, tn, fp, fn


def display_confusion_matrix():
    confusion_matrix_dict: dict = load_json(load_from_path=TEST_FINAL_CONFUSION_MATRIX_PATH)
    # 字典两个层次的key
    first_level = ["diff-mod", "vis", "infrared"]
    second_level = ["tp", "tn", "fp", "fn"]

    for key_1 in first_level:
        first_dict = confusion_matrix_dict.get(key_1)
        tp, tn, fp, fn = [first_dict.get(key_2) for key_2 in second_level]
        matrix = np.array(
            [
                [tp, tn],
                [fp, fn]
            ]
        )
        df_cm = pd.DataFrame(
            data=matrix,
            index=['True', 'False'],
            columns=['Pos', 'Neg']
        )
        # 绘制热力图
        sns.heatmap(
            df_cm,
            vmin=0,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        current_model_name, current_detector_name, *other_items = Path(TEST_FINAL_CONFUSION_MATRIX_PATH).stem.split("_")
        plt_title_name = f"CM_{current_model_name}_{current_detector_name}_{key_1}"
        plt.title(plt_title_name)
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig(
            f"{TEST_LOG_PATH}/{plt_title_name}.png",
            dpi=300,
            bbox_inches='tight',
        )
        plt.show()


def build_dataset_and_test_obj():
    my_dataset = build_dataset_for_test()
    my_test_obj = MyTestObj(dataset_for_test_obj=my_dataset)
    return my_dataset, my_test_obj


def read_detailed_result_json_file_to_get_confusion_matrix():
    """
    Read the result recorder JSON file, analyze the prediction results, and generate a confusion matrix.

    Parameters:
    None

    Returns:
    None

    This function reads the result recorder JSON file, which contains the prediction results of a model on a dataset.
    It then analyzes these results using the `analyze_result` function, which categorizes the results based on
    modalities (infrared, visible, or mixed) and computes the confusion matrix values (true positives, true negatives,
    false positives, and false negatives) for each category. Finally, the function saves the confusion matrix values
    to a new JSON file named `FINAL_CONFUSION_MATRIX_PATH` which is defined in a000_CONFIG.py.
    """
    result_lst = load_json(load_from_path=TEST_RESULT_RECORDER_PATH)
    confusion_matrix_dict = analyze_result(result_lst)
    save_to_json(
        obj=confusion_matrix_dict,
        save_to_path=TEST_FINAL_CONFUSION_MATRIX_PATH,
    )
