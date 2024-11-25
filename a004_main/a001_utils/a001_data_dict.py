import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

from a004_main.a001_utils.a000_CONFIG import TRAINING_VALI_SET_RATIO


class DatasetDictObj:
    def __init__(self, init_with_dataset_path=None, init_with_dict=None):
        if init_with_dict and init_with_dataset_path:
            raise ValueError("请指定init_with_dict或init_with_dataset_path的其中一个，不能同时使用。")
        if not init_with_dict and not init_with_dataset_path:
            raise ValueError("请指定init_with_dict或init_with_dataset_path的其中一个，不能都不指定。")

        if init_with_dataset_path:
            self.dataset_path = init_with_dataset_path
            self.dataset_dict = self.build_dataset_dict()
        else:
            self.dataset_path = None
            self.dataset_dict = init_with_dict

        # 获取所有人的key, key类型为字符串
        self.person_keys = list(self.dataset_dict.keys())
        # 总人数
        self.num_persons = len(self.person_keys)

        self.modalities_dict = {
            "mod_0": "infrared",
            "mod_1": "vis",
        }

    def build_dataset_dict(self) -> dict:
        """
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
        dataset_type_list: list = ["train", "val", "test"]
        original_modality_folder_name_list: list = ["gray", "rgb"]
        dataset_dict_0: dict = dict()
        for root, dirs, files in os.walk(self.dataset_path):
            root_path_obj = Path(root)
            root_path_parts: tuple = root_path_obj.parts
            # print(root_path_parts)

            # 从全部数据集中筛选需要的模态部分，不用其他多余的模态
            if root_path_obj.name == Path(self.dataset_path).name:
                for i in range(len(dirs) - 1, -1, -1):
                    dir_i = dirs[i]
                    if dir_i not in original_modality_folder_name_list:
                        del dirs[i]  # 必须原地修改dirs列表
                continue  # 如果进入了该if，则处理dirs之后直接去搜索下一个os.walk()

            # 注意root_path_parts[-1]就是root_path_parts.name
            if root_path_parts[-1] == "images" and root_path_parts[-2] in dataset_type_list:
                for image_path in files:
                    person_id = Path(image_path).stem.split("_")[0]
                    person_id = person_id.zfill(3)
                    person_key_in_dict = f"person_{person_id}"

                    # dict的get方法，获取指定key的value，如果key不存在返回None
                    # 利用返回None判断key是否存在，若不存在则添加
                    if dataset_dict_0.get(person_key_in_dict) is None:
                        dataset_dict_0[person_key_in_dict] = dict()
                    if dataset_dict_0[person_key_in_dict].get("infrared") is None:
                        dataset_dict_0[person_key_in_dict]["infrared"] = list()
                    if dataset_dict_0[person_key_in_dict].get("vis") is None:
                        dataset_dict_0[person_key_in_dict]["vis"] = list()

                    # 通过路径的倒数某节，获取这张图片是什么模态，然后重命名一下模态的叫法
                    modality_of_this_image = Path(root).parts[-3]
                    if modality_of_this_image == "gray":
                        modality_of_this_image = "infrared"
                    if modality_of_this_image == "rgb":
                        modality_of_this_image = "vis"

                    # 将图片路径添加到data_dict
                    full_path = str(Path(root) / Path(image_path))
                    dataset_dict_0[person_key_in_dict][modality_of_this_image].append(full_path)
        sorted_dataset_dict = {key: dataset_dict_0[key] for key in sorted(dataset_dict_0)}
        return sorted_dataset_dict

    def sample_an_image_given_mod_and_person_key(self, person_key, mod_choice) -> str:
        """
        从指定的人和模态，随机抽取一张图片
        Args:
            person_key (str)
            mod_choice (str)
        Returns:
            str
        """
        return random.choice(self.dataset_dict[person_key][mod_choice])

    def sample_two_images_from_same_person_given_mod(
            self,
            person_key,
            mod_choice_0,
            mod_choice_1
    ) -> tuple[str, str]:
        """
        从同一个人中抽取两张不同的图片，可以是跨模态或同模态
        """
        img0_path = random.choice(self.dataset_dict[person_key][mod_choice_0])
        img1_path = random.choice(self.dataset_dict[person_key][mod_choice_1])

        # 确保两张图片不同
        while img0_path == img1_path:
            img1_path = random.choice(self.dataset_dict[person_key][mod_choice_1])

        return img0_path, img1_path

    def split_dict_obj_to_training_and_vali(self):
        person_keys_training, person_keys_vali = train_test_split(
            self.person_keys,
            test_size=TRAINING_VALI_SET_RATIO,
            shuffle=True,
        )

        training_data_dict = {k: self.dataset_dict[k] for k in person_keys_training}
        vali_data_dict = {k: self.dataset_dict[k] for k in person_keys_vali}

        training_data_dict_obj = DatasetDictObj(init_with_dict=training_data_dict)
        vali_data_dict_obj = DatasetDictObj(init_with_dict=vali_data_dict)
        return training_data_dict_obj, vali_data_dict_obj
