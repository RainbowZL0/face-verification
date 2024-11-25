import random
import warnings
from itertools import accumulate
from pathlib import Path

import cv2
import numpy as np
from colorama import Fore
from deepface import DeepFace
from torch.utils.data import Dataset
from tqdm import tqdm

from a004_main.a001_utils.a000_CONFIG import VALI_SAMPLES_NUM, LOGGER
from a004_main.a001_utils.a001_data_dict import DatasetDictObj
from a004_main.a001_utils.a002_general_utils import (
    save_to_json,
)
from a004_main.a002_batch_test.a002_DatasetForTestOrVali import DatasetForTestOrVali


class DatasetForTrainingAndVali(Dataset):
    def __init__(
            self,
            original_dataset_path,
            num_samples_per_epoch,
            create_or_exist_cropped_dataset_at_path,
            training_detector_name,
            probability_for_mod_choices_for_training_dict=None,
            whether_build_cropped_dataset=False,
    ):
        """
        本类将成为MyTrainingObj类的成员属性，被维护。
        本类维护的重要属性：
        1. 还未区分Training和Validation的DatasetDictObj，包含全部数据。
        2. 用于Training的DatasetDictObj，是从全部数据分了一块。
        注意，不维护用于Vali的DatasetDictObj，使用split_out_dataset_for_vali()方法将它返回。
        Args:
            probability_for_mod_choices_for_training_dict: 默认值None将会触发使用函数内部定义的初始值，因为字典不建议直接作为默认参数值。
                一个例子如下，
                {
                    "diff-mod": 0.7,
                    "vis": 0.2,
                    "infrared": 0.1,
                }
                diff-mod：正例或反例中有任意一个与锚点模态不同，共六种，vii, viv, vvi; ivi, ivv, iiv
                vis：三元组都是vis, 即vvv
                infrared：三元组都是infrared, 即iii
        """
        """复制传入参数为属性"""
        self.original_dataset_path = original_dataset_path
        self.num_samples_per_epoch = num_samples_per_epoch
        self.create_or_exist_cropped_dataset_at_path = create_or_exist_cropped_dataset_at_path
        self.training_detector_name = training_detector_name
        self.probability_for_mod_choices_for_training_dict = probability_for_mod_choices_for_training_dict
        self.whether_build_cropped_dataset = whether_build_cropped_dataset

        """其他处理"""
        # 构造原始数据集字典
        self.original_dataset_dict_obj = DatasetDictObj(
            init_with_dataset_path=original_dataset_path,
        )

        # 根据传入参数whether_build_cropped_dataset做操作
        self.__build_cropped_dataset_files_if_need()
        # 确保cropped_dataset文件存在后，为其创建一个DatasetDictObj，真正用于训练和验证使用
        self.cropped_dataset_dict_obj = DatasetDictObj(
            init_with_dataset_path=self.create_or_exist_cropped_dataset_at_path,
        )
        self.cropped_dataset_for_training_dict_obj = None

        # 采样多模态的概率设定
        self.__check_and_init_probability_for_mod_choices_for_training_dict()
        # 概率字典确定后，生成一个accumulative概率的阶梯状表示，便于编写采样的代码
        self.accumulative_sum_of_probs_list: list = self.__get_accumulative_sum_of_probs_list()

    def split_out_dataset_for_vali(self) -> DatasetForTestOrVali:
        """
        示例初始化后，必须通过该方法获得vali的部分。
        """
        # 划分训练验证集的dataset dict
        (self.cropped_dataset_for_training_dict_obj,
         cropped_dataset_for_vali_dict_obj) = (
            self.cropped_dataset_dict_obj.split_dict_obj_to_training_and_vali())
        # training dataset dict保持原状。
        # validation dataset dict需要再被封装为dataset类.
        cropped_dataset_for_vali_obj = DatasetForTestOrVali(
            num_samples_per_epoch=VALI_SAMPLES_NUM,
            dataset_path=None,
            dataset_dict_obj=cropped_dataset_for_vali_dict_obj,
        )
        return cropped_dataset_for_vali_obj

    def __build_cropped_dataset_files_if_need(self):

        if self.whether_build_cropped_dataset:
            LOGGER.info(
                Fore.GREEN +
                "将要调用build_cropped_dataset()方法，如果你不需要这一步，"
                "请在构造DatasetForTraining对象时传入whether_build_cropped_dataset=False。"
            )
            self.__build_cropped_dataset()
        else:
            LOGGER.info(
                Fore.GREEN +
                f"Using existing cropped dataset at {self.create_or_exist_cropped_dataset_at_path}."
            )

    def __len__(self):
        return self.num_samples_per_epoch

    def __getitem__(self, index):
        """
        作为参考，dataset_for_test_or_val类返回的字典格式如下，
        {
            "img0_path": img0_path,
            "img1_path": img1_path,
            "person0_key": person0_key,
            "person1_key": person1_key if not is_positive else person0_key,
            "mod0": mod_choice_0,
            "mod1": mod_choice_1,
            "same_modality": True if mod_choice_0 == mod_choice_1 else False,
            "label": is_positive,
        }
        据此，决定本对象的返回字典样式如下，
        {
            "img_anchor_path": "...",
            "img_pos_path": "...",
            "img_neg_path": "...",
            "person_key_anchor_and_pos": "",
            "person_key_neg": "",
            "roll_mod_kind": 选项有 "diff-mod", "vis", "infrared"，选其一
            "specific_triplet_mods": 形如"iii"或"ivv"等等，分别代表三元组的模态,
            "mod_anchor": 选项有"infrared"或"vis"，选其一,
            "mod_pos": 同上,
            "mod_neg": 同上,
        }
        """
        # roll_mod_kind 选项有 "diff-mod", "vis", "infrared"
        roll_mod_kind = self.__roll_a_mod_kind_for_a_triplet_using_set_prob()
        # specific_mods形如"iii"或"ivv"等等。
        specific_triplet_mods: str = roll_specific_mods_given_kind(roll_mod_kind)
        # 转换表示形式为"infrared"或"vis"
        mod_anchor, mod_pos, mod_neg = [
            convert_str_form_from_single_char_to_full_name_for_mod(char_form=char_form)
            for char_form in specific_triplet_mods
        ]

        # 采样person_key，anchor和pos是同一个人，neg是另一个人
        person_key_anchor_and_pos, person_key_neg = self.__sample_person_key_for_anchor_pos_neg()

        # 采样图片路径
        img_anchor_path, img_pos_path = self.__sample_anchor_and_pos_image_path(
            person_key_anchor_and_pos=person_key_anchor_and_pos,
            mod_anchor=mod_anchor,
            mod_pos=mod_pos,
        )
        img_neg_path = self.__sample_neg_image_path(
            person_key_neg=person_key_neg,
            mod_neg=mod_neg,
        )

        return {
            "img_anchor_path": img_anchor_path,
            "img_pos_path": img_pos_path,
            "img_neg_path": img_neg_path,
            "person_key_anchor_and_pos": person_key_anchor_and_pos,
            "person_key_neg": person_key_neg,
            "roll_mod_kind": roll_mod_kind,
            "specific_triplet_mods": specific_triplet_mods,
            "mod_anchor": mod_anchor,
            "mod_pos": mod_pos,
            "mod_neg": mod_neg,
        }

    def __sample_anchor_and_pos_image_path(self, person_key_anchor_and_pos, mod_anchor, mod_pos):
        self.__check_training_part_is_not_none()
        img_anchor_path, img_pos_path = (
            self.cropped_dataset_for_training_dict_obj.sample_two_images_from_same_person_given_mod(
                person_key=person_key_anchor_and_pos,
                mod_choice_0=mod_anchor,
                mod_choice_1=mod_pos,
            )
        )
        return img_anchor_path, img_pos_path

    def __sample_neg_image_path(self, person_key_neg, mod_neg):
        self.__check_training_part_is_not_none()
        return self.cropped_dataset_for_training_dict_obj.sample_an_image_given_mod_and_person_key(
            person_key=person_key_neg,
            mod_choice=mod_neg,
        )

    def __check_training_part_is_not_none(self):
        """
        如果该属性为None，说明忘记划分training和vali，需要调用方法 split_out_dataset_for_vali
        """
        if self.cropped_dataset_for_training_dict_obj is None:
            raise ValueError(
                "DatasetForTraining实例的成员属性cropped_dataset_for_training_dict_obj在访问时仍然"
                "为None，没有经过初始化，说明忘记划分训练和验证集。请在实例化后调用split_out_dataset_for_vali"
                "方法解决。"
            )

    def __check_and_init_probability_for_mod_choices_for_training_dict(self):
        """
        如果probability_for_mod_choices_dict为None，没有传入值，则使用如下初始值.
        如果有传入值，需要检查是否概率和为1.
        """
        if self.probability_for_mod_choices_for_training_dict is None:
            LOGGER.info(
                Fore.GREEN +
                "未传入probability_for_mod_choices_dict，将使用初始值。")
            self.probability_for_mod_choices_for_training_dict = {
                "diff-mod": 0.7,
                "vis": 0.2,
                "infrared": 0.1,
            }
        else:
            if sum(self.probability_for_mod_choices_for_training_dict.values()) != 1:
                raise ValueError("probability_for_mod_choices_dict values must sum up to 1.")
        print("使用probability_for_mod_choices_dict:", self.probability_for_mod_choices_for_training_dict)

    def __build_cropped_dataset(self):
        """
        将原始数据集做人脸detection，然后保存到一个新目录。未识别到人脸的图片路径可在json中查看，然后手动标注。
        Returns:
            字典，包含所有数据，一个示例的结构层次如下。
            data_dict = {
                "person_0": {
                    "infrared": ["path/to/infrared1.png", "path/to/infrared2.png"],
                    "vis": ["path/to/vis1.png", "path/to/vis2.png"]
                },
                "person_1": {
                    "infrared": ["path/to/infrared3.png", "path/to/infrared4.png"],
                    "vis": ["path/to/vis3.png", "path/to/vis4.png"]
                },
                # 其他人...
            }
        """
        # 如果还没有文件夹保存识别到的人脸图片，则创建文件夹
        sf_tl54_cropped_folder_path = Path(self.create_or_exist_cropped_dataset_at_path)
        if not sf_tl54_cropped_folder_path.exists():
            sf_tl54_cropped_folder_path.mkdir(parents=True, exist_ok=True)
        else:
            LOGGER.warn(
                Fore.LIGHTRED_EX +
                f"Folder {self.create_or_exist_cropped_dataset_at_path} already exists.\n"
                f"Maybe you do not need to run this function for face detection again.\n"
                f"Please check twice to make sure it is really what you want.\n"
            )
        sf_tl54_cropped_folder_name = sf_tl54_cropped_folder_path.name

        # 把所有图片路径都收集到一个没有层次的大列表里
        dataset_dict = self.original_dataset_dict_obj.dataset_dict
        dataset_list = list()
        for data_dict_item in dataset_dict.values():
            # dataset_list_item: {
            #     "infrared": ["path/to/infrared1.png", "path/to/infrared2.png", ...],
            #     "vis": ["path/to/vis1.png", "path/to/vis2.png", ...],
            # }
            for path_list in data_dict_item.values():
                dataset_list.extend(path_list)

        # 适用于批量推理的代码，但deepface的extract_faces()函数不支持batch推理，先做注释。
        # num_iters = ceil(len(dataset_list) / batch_size)
        # for i in range(num_iters):
        #     batch_start = i * batch_size
        #     batch_end = min((i + 1) * batch_size, len(dataset_list))
        #     batch_data_list = dataset_list[batch_start:batch_end]
        #
        #     DeepFace.extract_faces(batch_data_list)

        # for debug use, show detection result with matplotlib
        # figure, axes = init_a_figure_and_an_axes()

        failed_image_path_list = []

        for img_path in tqdm(dataset_list):
            try:
                face_dict_list = DeepFace.extract_faces(
                    img_path=cv2.imread(img_path),
                    detector_backend=self.training_detector_name,
                    enforce_detection=False,
                )
            except Exception as e:
                LOGGER.warn(
                    Fore.LIGHTRED_EX +
                    f"在该图片上没有发现人脸：{img_path}.\n"
                    f"Exception: {e}"
                )
                failed_image_path_list.append(img_path)
                continue
            else:
                # the_first_face: dict {
                #   "face": ndarray,  # 注意格式是 BGR float HxWxC
                #   "facial_area": {},
                #   "confidence": float,
                # }
                the_first_face = face_dict_list[0]["face"]

                # 检查是否创建了图片对应路径的文件夹，如果没有则创建
                img_path_obj = Path(img_path)
                cropped_img_folder_path, img_cropped_name = img_path_obj.parent, Path(img_path_obj.name)
                img_parent_folder_path_level_list = list(cropped_img_folder_path.parts)
                img_parent_folder_path_level_list[3] = sf_tl54_cropped_folder_name
                img_cropped_folder_path = Path(*img_parent_folder_path_level_list)
                img_cropped_path = img_cropped_folder_path / img_cropped_name

                if not img_cropped_folder_path.exists():
                    img_cropped_folder_path.mkdir(parents=True, exist_ok=True)
                if img_cropped_path.exists():
                    warnings.warn(
                        f"The cropped image {img_cropped_path} already exists.\n"
                        f"Please check whether you have created it twice as undancy.\n"
                    )

                if the_first_face.dtype == np.float64:
                    LOGGER.warn(
                        Fore.LIGHTRED_EX +
                        f"The image will immediately be converted from {the_first_face.dtype} to {np.uint8}\n"
                        f"to support cv2.cvtColor() and cv2.imwrite().\n"
                    )
                    the_first_face *= 255
                    the_first_face = the_first_face.astype(np.uint8)
                the_first_face = cv2.cvtColor(src=the_first_face, code=cv2.COLOR_BGR2RGB)
                # the_first_face = cv2.normalize(
                #     src=the_first_face,
                #     dst=None,
                #     alpha=0,
                #     beta=255,
                #     norm_type=cv2.NORM_MINMAX,
                #     dtype=cv2.CV_8U,
                # )

                # for debug use, show the detection result with matplotlib
                # adjust_figure_size_and_show_image_and_release_resources(
                #     img=the_first_face,
                #     figure=figure,
                #     axes=axes
                # )

                cv2.imwrite(filename=str(img_cropped_path), img=the_first_face)
                print(
                    f"Cropped image saved at {str(img_cropped_path)}.\n"
                )

        print(
            f"Processing done.\n"
            f"Failed to detect faces in {len(failed_image_path_list)} images.\n"
            f"Details will be saved in json file.\n"
        )
        save_json_path = (Path(self.create_or_exist_cropped_dataset_at_path) /
                          Path("detection_failed_images_path_list.json"))
        save_to_json(failed_image_path_list, save_json_path)

    def __sample_for_a_triplet(
            self,
            person_key__for_image_0_and_1: str,
            person_key__for_image_2: str,
            mod_choices_list: tuple | list,
    ) -> tuple[str, str, str]:
        """
        给定mod_choices, 返回anchor, positive, negative三图片的路径。
        实现：前两图来自同一个人，可以直接调用dataset_dict_obj的方法sample_two_images_from_same_person_given_mod()。
        最后一张图调用dataset_dict_obj的sample_an_image_given_mod_and_person_key()方法。
        Args:
            person_key__for_image_0_and_1: .
            person_key__for_image_2: .
            mod_choices_list: 形如["infrared", "vis", "infrared"]的三个模态选择。
        Returns:
            tuple of three images paths.
        """
        anchor_img_path, positive_img_path = (
            self.original_dataset_dict_obj.sample_two_images_from_same_person_given_mod(
                person_key=person_key__for_image_0_and_1,
                mod_choice_0=mod_choices_list[0],
                mod_choice_1=mod_choices_list[1],
            )
        )

        # 获取最后一张图片的路径
        negative_img_path = self.original_dataset_dict_obj.sample_an_image_given_mod_and_person_key(
            person_key=person_key__for_image_2,
            mod_choice=mod_choices_list[2],
        )

        return anchor_img_path, positive_img_path, negative_img_path

    def __roll_a_mod_kind_for_a_triplet_using_set_prob(self) -> str:
        """
        self.probability_for_mod_choices_dict: 一个例子如下，
            {
                "diff-mod": 0.7,
                "vis": 0.2,
                "infrared": 0.1,
            }
            diff-mod：正例或反例中有任意一个与锚点模态不同，共六种，vii, viv, vvi; ivi, ivv, iiv
            vis：三元组都是vis, 即vvv
            infrared：三元组都是infrared, 即iii
        Returns:
            str，三元组种类，在"diff-mod", "vis", "infrared"中选其一。
        """
        roll_dice = random.random()
        if self.accumulative_sum_of_probs_list[0] <= roll_dice < self.accumulative_sum_of_probs_list[1]:
            mod_kind = "diff-mod"
        elif self.accumulative_sum_of_probs_list[1] <= roll_dice < self.accumulative_sum_of_probs_list[2]:
            mod_kind = "vis"
        else:
            mod_kind = "infrared"
        return mod_kind

    def __get_accumulative_sum_of_probs_list(self) -> list:
        # 根据传入的概率设定，决定是抽样到diff-mod还是vis还是infrared
        # 使用示例的传入dict，构造accumulate_prob = [0, 0.7, 0.9, 1.0]
        accumulate_prob: list = list(accumulate(self.probability_for_mod_choices_for_training_dict.values()))
        accumulate_prob.insert(0, 0.0)
        return accumulate_prob

    def __sample_person_key_for_anchor_pos_neg(self):
        self.__check_training_part_is_not_none()
        person_key_anchor_pos, person_key_neg = random.sample(
            population=self.cropped_dataset_for_training_dict_obj.person_keys,
            k=2,
        )
        return person_key_anchor_pos, person_key_neg


def roll_specific_mods_given_kind(mod_kind: str) -> str:
    """
    Args:
        mod_kind: 可能是"diff-mod", "vis", "infrared"中的一种。
    Returns:
        形如"iii"的三元组模态选择。
    """
    if mod_kind == "diff-mod":
        return roll_specific_mods_for_diff_mod()
    elif mod_kind == "vis":
        return roll_specific_mods_for_vis()
    elif mod_kind == "infrared":
        return roll_specific_mods_for_infrared()
    else:
        raise ValueError(f"Invalid mod_kind: {mod_kind}")


def roll_specific_mods_for_diff_mod():
    mod_choices = ["vii", "viv", "vvi", "ivv", "ivi", "iiv"]
    return random.choice(mod_choices)


def roll_specific_mods_for_vis():
    return "vvv"


def roll_specific_mods_for_infrared():
    return "iii"


def convert_str_form_from_single_char_to_full_name_for_mod(char_form: str):
    if char_form == "i":
        return "infrared"
    elif char_form == "v":
        return "vis"
    else:
        raise ValueError(f"Invalid char_form for mod: {char_form}, it should be 'i' or 'v'.")


if __name__ == '__main__':
    pass
