import random

from torch.utils.data import Dataset

from a004_main.a001_utils.a001_data_dict import DatasetDictObj


class DatasetForTestOrVali(Dataset):
    def __init__(
            self,
            num_samples_per_epoch,
            dataset_path=None,
            dataset_dict_obj: DatasetDictObj = None
    ):
        """
        提供两种构造方式，用dataset_path，或直接传入已有的dataset_dict_obj
        Args:
            num_samples_per_epoch: 每轮epoch采样多少对图片
            dataset_path: 使用的数据集所在目录
        """
        # 每轮epoch采样多少对图片
        self.num_samples_per_epoch = num_samples_per_epoch

        if dataset_path is not None and dataset_dict_obj is None:
            # 使用的数据集所在目录
            self.dataset_path = dataset_path
            # data_dict 是包含每个人红外和可见光图片路径的字典
            self.dataset_dict_obj = DatasetDictObj(
                init_with_dataset_path=dataset_path,
            )
        elif dataset_path is None and dataset_dict_obj is not None:
            self.dataset_path = dataset_dict_obj.dataset_path
            self.dataset_dict_obj = dataset_dict_obj
        else:
            raise ValueError(
                f"构造DatasetForTest对象时必须只传入dataset_path或dataset_dict_obj中的一个。"
                f"如果都不传入，或都传入，都会导致错误。现在你传入的是dataset_path={dataset_path}, "
                f"dataset_dict_obj={dataset_dict_obj}。"
            )

        self.modality_choices_for_test_dict = self.get_modality_choices_for_test_dict()

    def __len__(self):
        # 根据需求定义数据集的长度
        return self.num_samples_per_epoch  # 假设定义1000次采样

    def __getitem__(self, idx):
        # 随机决定是正例还是反例。
        is_positive = random.choice([True, False])
        # 随机抽取一对模态
        mod_choice_0, mod_choice_1 = self.get_a_pair_of_random_modality_choice()
        # 初始化被选中人的key
        person0_key, person1_key = str(), str()

        if is_positive:
            # 正例：从同一个人中抽取两张不同的图片
            person0_key = random.choice(self.dataset_dict_obj.person_keys)
            img0_path, img1_path = self.dataset_dict_obj.sample_two_images_from_same_person_given_mod(
                person_key=person0_key,
                mod_choice_0=mod_choice_0,
                mod_choice_1=mod_choice_1,
            )
        else:
            # 反例：从两个不同的人中抽取图片
            person0_key, person1_key = random.sample(
                self.dataset_dict_obj.person_keys,
                2
            )
            img0_path = self.dataset_dict_obj.sample_an_image_given_mod_and_person_key(
                person_key=person0_key,
                mod_choice=mod_choice_0,
            )
            img1_path = self.dataset_dict_obj.sample_an_image_given_mod_and_person_key(
                person_key=person1_key,
                mod_choice=mod_choice_1,
            )

        # 返回采样的两张图片路径等信息
        result_dict = {
            "img0_path": img0_path,
            "img1_path": img1_path,
            "person0_key": person0_key,
            "person1_key": person1_key if not is_positive else person0_key,
            "mod0": mod_choice_0,
            "mod1": mod_choice_1,
            "same_modality": True if mod_choice_0 == mod_choice_1 else False,
            "label": is_positive,
        }
        return result_dict

    def get_modality_choices_for_test_dict(self) -> dict:
        mod_0, mod_1 = (self.dataset_dict_obj.modalities_dict["mod_0"],
                        self.dataset_dict_obj.modalities_dict["mod_1"])
        modality_choices_dict: dict = {
            "choice_0": [mod_0, mod_0],
            "choice_1": [mod_0, mod_1],
            "choice_2": [mod_1, mod_1],
        }
        return modality_choices_dict

    def get_a_pair_of_random_modality_choice(self) -> tuple:
        mod_choice_list: list = random.choice(
            list(
                self.modality_choices_for_test_dict.values()
            )
        )
        return tuple(mod_choice_list)
