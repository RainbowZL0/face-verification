from pathlib import Path
from textwrap import dedent

import seaborn as sns
import torch
from colorama import Fore
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import TripletMarginWithDistanceLoss
from torch.nn.functional import cosine_similarity
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.transforms import v2
from tqdm import tqdm

from a004_main.a001_utils.a000_CONFIG import (
    TRAINING_INITIAL_LR,
    TRAINING_MINIMUM_LR,
    TRAINING_TOTAL_EPOCHS,
    TRAINING_SAVE_MODEL_INTERVAL_IN_EPOCHS,
    TRAINING_BATCH_SIZE,
    TRAINING_OR_VALI_DEVICE,
    TRAINING_PRINT_INFO_INTERVAL_IN_ITERS,
    VALI_BATCH_SIZE,
    TRAINING_SAVE_MODEL_TO_FOLDER,
    LOAD_FROM_STATE_PATH,
    WHETHER_USING_SAVED_STATE,
    TRAINING_VALI_INTERVAL_IN_ITERS,
    VALI_LOG_FOLDER,
    LOGGER,
    LOSS_FUNC_MARGIN,
    LOSS_FUNC_SWAP,
    LOSS_FUNC_WEIGHT_D_AN_PENALTY,
    LOSS_FUNC_USING_SELF_DEFINED,
)
from a004_main.a001_utils.a002_general_utils import (
    my_collate_fn_factory,
    get_time_str,
    save_to_json,
    load_json,
    loss_penalty_func_for_d_an,
    my_distance_func
)
from a004_main.a003_training.a002_DatasetForTraining import DatasetForTrainingAndVali


class MyTrainingObj:
    def __init__(self, dataset_for_training: DatasetForTrainingAndVali):
        self.dataset_for_training = dataset_for_training
        self.dataset_for_vali = self.dataset_for_training.split_out_dataset_for_vali()
        self.dataloader_for_training = DataLoader(
            dataset=self.dataset_for_training,
            batch_size=TRAINING_BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            collate_fn=my_collate_fn_factory(batch_size=TRAINING_BATCH_SIZE),
        )
        self.dataloader_for_vali = DataLoader(
            dataset=self.dataset_for_vali,
            batch_size=VALI_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=my_collate_fn_factory(batch_size=VALI_BATCH_SIZE),
        )
        # 为train和vali封装两个不同的transform，区别是训练需要随机数据增强。
        self.transform_for_training = _get_transform(whether_use_augmentation_for_training=True)
        self.transform_for_vali = _get_transform(whether_use_augmentation_for_training=False)
        # 定义模型
        self.model = InceptionResnetV1(pretrained="vggface2").to(device=TRAINING_OR_VALI_DEVICE)
        # 优化器和学习率调整
        self.optimizer = optim.Adam(self.model.parameters(), lr=TRAINING_INITIAL_LR)
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=TRAINING_TOTAL_EPOCHS,
            eta_min=TRAINING_MINIMUM_LR,
        )
        # 损失函数
        if LOSS_FUNC_USING_SELF_DEFINED:
            self.loss_func = try_defining_a_better_loss_func
        else:
            self.loss_func = TripletMarginWithDistanceLoss(
                distance_function=my_distance_func,
                margin=LOSS_FUNC_MARGIN,
                swap=LOSS_FUNC_SWAP,
            )

        # tensorboard
        self.tensorboard_writer = SummaryWriter()

        # iter和epoch记录
        self.current_epochs = 0  # 取值范围 [0, 总epoch数)
        self.current_epochs_float = 0.0
        self.current_iters_in_an_epoch = 1  # 取值范围 [1, 一个epoch内的iter数]
        self.iters_up_to_now = 1

        self.detailed_vali_result_list = []

    def start_train_and_vali(self):
        if WHETHER_USING_SAVED_STATE:
            self.load_my_state()
        else:
            LOGGER.info(
                Fore.GREEN +
                "开始从头训练，没有加载任何状态。如果需要加载，请在Config中设置WHETHER_USING_SAVED_STATE为True。"
            )
        LOGGER.info(
            Fore.GREEN +
            f"Starting at epoch = {self.current_epochs}, "
            f"using lr = {self.scheduler.get_last_lr()[0]}, "
            f"iters in one epoch = {self.current_iters_in_an_epoch}"
        )
        for epoch in tqdm(range(self.current_epochs, TRAINING_TOTAL_EPOCHS)):
            self.current_epochs = epoch
            self.__train_for_one_epoch_with_vali()
            if (epoch + 1) % TRAINING_SAVE_MODEL_INTERVAL_IN_EPOCHS == 0:
                self.save_my_state()
        self.tensorboard_writer.close()

    def __train_for_one_epoch_with_vali(self):
        self.model.train()
        for current_iter, data_dict in tqdm(enumerate(self.dataloader_for_training, start=1)):
            data_dict: dict
            # batch_size为1时data_dict如下。batch更大时，每个value变为list。
            # {
            #   "img_anchor_path": "...",
            #   "img_pos_path": "...",
            #   "img_neg_path": "...",
            #   "person_key_anchor_and_pos": "",
            #   "person_key_neg": "",
            #   "roll_mod_kind": 选项有 "diff-mod", "vis", "infrared"，选其一
            #   "specific_triplet_mods": 形如"iii"或"ivv"等等，分别代表三元组的模态,
            #   "mod_anchor": 选项有"infrared"或"vis"，选其一,
            #   "mod_pos": 同上,
            #   "mod_neg": 同上,
            # }

            # 记录当前iter数量
            self.current_iters_in_an_epoch = current_iter
            self.iters_up_to_now = self.__calcu_iters_up_to_now()
            # 读取图片为tensor。每个变量形状都是(batch C H W)。
            anchor_batch, pos_batch, neg_batch = (
                _read_paths_to_tensor_and_transform(
                    path_or_path_list=data_dict[key],
                    using_transform=self.transform_for_training,
                )
                for key in ["img_anchor_path", "img_pos_path", "img_neg_path"]
            )
            anchor_batch: torch.Tensor
            pos_batch: torch.Tensor
            neg_batch: torch.Tensor

            # forward
            anchor_batch, pos_batch, neg_batch = (
                self.model(batch) for batch in [anchor_batch, pos_batch, neg_batch]
            )

            # loss
            if LOSS_FUNC_USING_SELF_DEFINED:
                loss_dict = self.loss_func(anchor_batch, pos_batch, neg_batch)
                loss = loss_dict['loss']
            else:
                loss = self.loss_func(anchor_batch, pos_batch, neg_batch)
                d_ap = my_distance_func(anchor_batch, pos_batch).mean()
                d_an = my_distance_func(anchor_batch, neg_batch).mean()
                d_pn = my_distance_func(pos_batch, neg_batch).mean()
                if LOSS_FUNC_SWAP and d_pn < d_an:
                    d_an = d_pn
                loss_dict = {
                    'loss': loss,
                    'd_ap': d_ap,
                    'd_an': d_an,
                }

            # print training loss info
            self.__print_training_info(loss.item())
            # tensorboard
            self.__submit_loss_to_tensorboard(
                loss_dict=loss_dict,
                lr=self.scheduler.get_last_lr()[0]
            )

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # lr scheduler
            self.current_epochs_float = self.__calcu_current_epochs_float()
            self.scheduler.step(self.current_epochs_float)  # type: ignore

            # 训练过程中，每指定的iters过后，进行validation
            if self.iters_up_to_now % TRAINING_VALI_INTERVAL_IN_ITERS == 0:
                detailed_result_list, detailed_result_list_save_to_json_path = self.vali()
                analyze_detailed_result_to_get_cosine_similarity_distribution(
                    detailed_result_list=detailed_result_list,
                )

    def vali(self):
        """
        遍历得到的batch_dict格式如下。
        {
            "img0_path": img0_path,
            "img1_path": img1_path,
            "person0_key": person0_key,
            "person1_key": person1_key if not is_positive else person0_key,
            "mod0": infrared or vis,
            "mod1": infrared or vis,
            "same_modality": True if mod_choice_0 == mod_choice_1 else False,
            "label": is_positive,
        }

        在之前写的测试类中，每一对验证结果的信息字典格式如下，比batch_dict多出两项。
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
        然而，我们在本函数中只能得到余弦相似度，还没有为其设定阈值，所以无法直接得到model_prediction_result和
        prediction_compared_with_label。不过为了统一格式，仍然带上这两项，取值为None。
        随后，额外加一项cosine_similarity。

        最终确定如下：
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
            'cosine_similarity': float -1 ~ 1,
            'cosine_distance': float 0 ~ 2,
        }
        """
        LOGGER.info(
            Fore.GREEN +
            "正在进行validation..."
        )
        # 模型训练状态维护
        whether_is_training = self.model.training  # 记录原本的训练状态
        self.model.eval()
        # 创建一个列表以记录所有样本的验证结果
        detailed_result_list = []
        # 开始验证
        with torch.no_grad():
            # 批量计算余弦相似度
            for batch_dict in tqdm(self.dataloader_for_vali):
                # 读取图片为tensor。每个变量形状都是(batch C H W)。
                img0_batch, img1_batch = (
                    _read_paths_to_tensor_and_transform(
                        path_or_path_list=batch_dict[key],
                        using_transform=self.transform_for_vali,
                    )
                    for key in ["img0_path", "img1_path"]
                )
                img0_batch_vector = self.model(img0_batch)
                img1_batch_vector = self.model(img1_batch)

                cosine_similarity_list = cosine_similarity(
                    x1=img0_batch_vector,
                    x2=img1_batch_vector,
                    dim=1,
                ).detach().cpu().tolist()
                cosine_distance_list = my_distance_func(
                    tensor_0=img0_batch_vector,
                    tensor_1=img1_batch_vector
                ).detach().cpu().tolist()

                if VALI_BATCH_SIZE > 1:
                    for i in range(len(cosine_similarity_list)):
                        part_1_of_result_dic = {
                            k: batch_dict[k][i] for k in batch_dict.keys()
                        }
                        part_2_of_result_dic = {
                            'model_prediction_result': None,
                            'prediction_compared_with_label': None,

                            'cosine_similarity': cosine_similarity_list[i],
                            'cosine_distance': cosine_distance_list[i],
                        }
                        result_dic = {
                            **part_1_of_result_dic,
                            **part_2_of_result_dic,
                        }
                        detailed_result_list.append(result_dic)
                else:
                    # means batch_size == 1
                    part_1_of_result_dic = batch_dict.copy()
                    part_2_of_result_dic = {
                        'model_prediction_result': None,
                        'prediction_compared_with_label': None,

                        'cosine_similarity': cosine_similarity_list[0],
                        'cosine_distance': cosine_distance_list[0]
                    }
                    result_dic = {
                        **part_1_of_result_dic,
                        **part_2_of_result_dic,
                    }
                    detailed_result_list.append(result_dic)
        # 保存为json到磁盘
        folder_path_obj = Path(VALI_LOG_FOLDER)
        if not folder_path_obj.exists():
            folder_path_obj.mkdir(parents=True, exist_ok=True)
        js_file_name_obj = Path(get_time_str() + ".json")
        js_path_obj = folder_path_obj / js_file_name_obj
        save_to_json(obj=detailed_result_list, save_to_path=js_path_obj)
        LOGGER.info(f"验证结果已输出到{str(js_path_obj)}")

        # 训练状态维护 恢复
        if whether_is_training:
            self.model.train()

        # 记录并返回关键结果
        if self.detailed_vali_result_list:
            # 清空上次的结果记录
            self.detailed_vali_result_list.clear()
        self.detailed_vali_result_list.extend(detailed_result_list)

        return detailed_result_list, js_path_obj

    def save_my_state(self):
        time_str = get_time_str()
        model_file_name = (f"{time_str}_epochs-{self.current_epochs + 1}_"
                           f"iters-up-to-now-{self.iters_up_to_now}.pth")
        save_model_to_folder_obj = Path(TRAINING_SAVE_MODEL_TO_FOLDER)
        save_state_to_path = str(save_model_to_folder_obj / Path(model_file_name))

        if not save_model_to_folder_obj.exists():
            save_model_to_folder_obj.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "current_epochs": self.current_epochs + 1,
            "current_iters_in_an_epoch": self.current_iters_in_an_epoch,
            "iters_up_to_now": self.iters_up_to_now,
        }
        torch.save(state, save_state_to_path)
        print(Fore.YELLOW + f"State saved to '{save_state_to_path}'")

    def __submit_loss_to_tensorboard(self, loss_dict, lr):
        for k, v in loss_dict.items():
            self.tensorboard_writer.add_scalar(
                tag=f"training/{k}",
                scalar_value=v.item() if isinstance(v, torch.Tensor) else v,
                global_step=self.iters_up_to_now,
                new_style=True,
            )
        self.tensorboard_writer.add_scalar(
            tag="lr",
            scalar_value=lr,
            global_step=self.iters_up_to_now,
            new_style=True,
        )
        self.tensorboard_writer.flush()

    def __calcu_iters_up_to_now(self):
        return self.current_epochs * len(self.dataloader_for_training) + self.current_iters_in_an_epoch

    def __calcu_current_epochs_float(self):
        return self.current_epochs + self.current_iters_in_an_epoch / len(self.dataloader_for_training)

    def __print_training_info(self, loss):
        if self.current_epochs % TRAINING_PRINT_INFO_INTERVAL_IN_ITERS == 0:
            LOGGER.info(dedent(
                Fore.GREEN +
                f"""
                Training Info: 
                Current epochs = {self.current_epochs + 1},
                Current epochs float = {self.current_epochs_float},
                Total epochs = {TRAINING_TOTAL_EPOCHS},
                Current iters in this epoch = {self.current_iters_in_an_epoch},
                Total iters in this epoch = {len(self.dataloader_for_training)},
                Iters up to now = {self.iters_up_to_now},
                Loss = {round(loss, 5)},
                Using LR = {self.scheduler.get_last_lr()[0]}
                """
            ))

    def load_my_state(self):
        """
        存储时的格式如下，
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "current_epochs": self.current_epochs + 1,
            "current_iters_in_an_epoch": self.current_iters_in_an_epoch,
            "iters_up_to_now": self.iters_up_to_now,
        }
        """
        print(Fore.CYAN + f"Loading state from '{LOAD_FROM_STATE_PATH}'")

        read_state = torch.load(
            LOAD_FROM_STATE_PATH,
            map_location=TRAINING_OR_VALI_DEVICE
        )

        # 分别load state
        self.model.load_state_dict(read_state["model_state"])
        self.optimizer.load_state_dict(read_state["optimizer_state"])
        self.scheduler.load_state_dict(read_state["scheduler_state"])
        self.current_epochs = read_state["current_epochs"]
        self.current_iters_in_an_epoch = read_state["current_iters_in_an_epoch"]
        self.iters_up_to_now = read_state["iters_up_to_now"]

        print(Fore.CYAN + f"State is loaded successfully.")


def analyze_detailed_result_to_get_cosine_similarity_distribution(
        detailed_result_list: list = None,
        detailed_result_json_path: Path or str = None,
):
    """
    传入两个参数其一，用于分析。

    list的每一项字典的格式如下，
    result_dict = {
        'img0_path': '..\\..\\a000_DATASET\\SF-TL54\\gray\\test\\images\\119_1_2_8_144_102_1.png',
        'img1_path': '..\\..\\a000_DATASET\\SF-TL54\\gray\\test\\images\\119_1_2_1_115_32_1.png',
        'person0_key': 'person_119',
        'person1_key': 'person_119',
        'mod0': 'infrared',
        'mod1': 'infrared',
        'same_modality': True,
        'label': True,
        'model_prediction_result': None,
        'prediction_compared_with_label': None,
        'cosine_similarity': float,
    }
    第一个中间步骤，把list拆成3个小list，按照模态diff-mod、infrared、vis
    第二步，为三个模态分别绘制正例、反例的cos_similarity分布图。每个模态使用一个figure，然后正反例分别做成两个axes。
    """
    if detailed_result_list is None and detailed_result_json_path is None:
        raise ValueError(
            "detailed_result_list 或 detailed_result_json_path 必须传入其一。当前状态：都未传入。"
        )
    elif detailed_result_list is not None and detailed_result_json_path is not None:
        raise ValueError(
            f"detailed_result_list 或 detailed_result_json_path 只能传入其一。"
            f"当前状态：detailed_result_list = {detailed_result_list}, "
            f"detailed_result_json_path = {detailed_result_json_path}"
        )
    else:
        if detailed_result_json_path is not None:
            result_list = load_json(detailed_result_json_path)
        else:
            result_list = detailed_result_list
        three_mod_kinds: dict = __split_sub_list_by_mod_kind(result_list)
        # three_mod_kinds = {
        #     'infrared': [],
        #     'diff-mod': [],
        #     'vis': [],
        # }
        for mod_kind, dict_list in three_mod_kinds.items():
            pos_neg_dict = __split_sub_list_by_pos_neg(dict_list)
            # pos_neg_dict = {
            #   'pos': [],
            #   'neg': [],
            # }
            __draw(mod_kind=mod_kind, pos_neg_dict=pos_neg_dict)


def __split_sub_list_by_mod_kind(result_list):
    split_list_by_mod_kind = {
        'infrared': [],
        'diff-mod': [],
        'vis': [],
    }
    for dic in result_list:
        mod_kind_and_also_the_dict_key = __choose_mod_kind_given_a_dict(dic)
        split_list_by_mod_kind[mod_kind_and_also_the_dict_key].append(dic)
    return split_list_by_mod_kind


def __split_sub_list_by_pos_neg(result_list):
    result_dic = {
        'pos': [],
        'neg': [],
    }
    for dic_item in result_list:
        if dic_item['label']:
            result_dic['pos'].append(dic_item)
        else:
            result_dic['neg'].append(dic_item)
    return result_dic


def __draw(mod_kind, pos_neg_dict):
    """
    pos_neg_dict: {
        'pos': [some result_dict],
        'neg': [some result_dict],
    }
    """
    pos_neg_data_dict = {
        'pos_cos_distance': [dic['cosine_distance'] for dic in pos_neg_dict['pos']],
        'neg_cos_distance': [dic['cosine_distance'] for dic in pos_neg_dict['neg']],
    }
    __handle_figure_and_2_axes(fig_title=mod_kind, pos_neg_data_dict=pos_neg_data_dict)


def __handle_figure_and_2_axes(fig_title, pos_neg_data_dict):
    """
    pos_neg_data_dict = {
        'pos_cos_distance': [],
        'neg_cos_distance': [],
    }
    """
    sns.set_theme(style="darkgrid")

    fig = plt.figure(dpi=300, layout='compressed', figsize=(20, 6))
    fig.suptitle(fig_title, fontsize=16)
    axes_array = fig.subplots(1, 2, sharex=True, sharey=True)

    sns.despine(fig)

    for index, (key, data) in enumerate(pos_neg_data_dict.items()):
        sns.histplot(
            data=data,
            kde=True,
            ax=axes_array[index],
            bins=70,
            stat='density',
            element='step',
        )
        axes_array[index].set_title(key)
        axes_array[index].set_xlabel('cosine_distance')
        axes_array[index].set_xlim(0, 2)

    fig.show()
    plt.close(fig)


def __choose_mod_kind_given_a_dict(result_dict):
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
        'model_prediction_result': None,
        'prediction_compared_with_label': None,
        'cosine_similarity': float,
    }
    """
    if result_dict["mod0"] == "infrared" and result_dict["mod1"] == "infrared":
        return 'infrared'
    elif result_dict["mod0"] == "infrared" and result_dict["mod1"] == "vis":
        return 'diff-mod'
    elif result_dict["mod0"] == "vis" and result_dict["mod1"] == "vis":
        return 'vis'
    else:
        raise ValueError("无效的mod0和mod1组合。")


def _read_paths_to_tensor_and_transform(path_or_path_list, using_transform):
    if isinstance(path_or_path_list, str):
        img = _read_an_image_and_transform(path=path_or_path_list, using_transform=using_transform)
        return img.unsqueeze(0).to(device=TRAINING_OR_VALI_DEVICE)  # 加上batch dim
    elif isinstance(path_or_path_list, list):
        tensor_list = list()
        for path in path_or_path_list:
            img = _read_an_image_and_transform(path=path, using_transform=using_transform)
            tensor_list.append(img)
        return torch.stack(tensor_list).to(device=TRAINING_OR_VALI_DEVICE)
    else:
        raise ValueError("Input should be a string or a list of strings.")


def _read_an_image_and_transform(path, using_transform):
    img = read_image(path)
    if img.shape[0] == 4:  # 如果是4通道图像RGBA，直接丢掉A，变为RGB
        img = img[:3, :, :]
    img = using_transform(img)
    return img


def _get_transform(whether_use_augmentation_for_training):
    list_for_compose = [
        v2.ToDtype(torch.float32),
        v2.Resize((160, 160)),
        # 这组norm的选择会将数值范围从0~255映射到-1~1，这是facenet-pytorch库使用的值，与它保持一致。
        v2.Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
    ]
    if whether_use_augmentation_for_training:
        list_for_compose.append(
            v2.RandomHorizontalFlip()
        )
    return v2.Compose(list_for_compose)


def try_defining_a_better_loss_func(
        anchor,
        pos,
        neg,
        distance_func: callable = my_distance_func,
):
    d_ap = distance_func(anchor, pos).mean()
    d_an = distance_func(anchor, neg).mean()
    d_pn = distance_func(pos, neg).mean()

    if LOSS_FUNC_SWAP and d_pn.item() < d_an.item():
        d_an = d_pn

    # loss: torch.Tensor = -d_an + LOSS_FUNC_MARGIN
    loss: torch.Tensor = d_ap - d_an + LOSS_FUNC_MARGIN

    # penalty for d_an
    penalty_value = (
            LOSS_FUNC_WEIGHT_D_AN_PENALTY *
            loss_penalty_func_for_d_an(d_an.item())
    )
    loss += penalty_value

    if loss.item() < 0:
        loss -= loss

    return {
        'loss': loss,
        'd_ap': d_ap.item(),
        'd_an': d_an.item(),
        'penalty_value': penalty_value,
    }
