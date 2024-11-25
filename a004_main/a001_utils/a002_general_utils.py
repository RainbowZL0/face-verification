import json
import math
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch.nn import functional

from a004_main.a001_utils.a000_CONFIG import *


# 已延迟导入 from a002_dataset_for_test import DatasetForTest


def my_collate_fn_factory(batch_size):
    if batch_size == 1:
        def my_collate_fn_for_batch_size_being_1(batch_dict_list):
            return batch_dict_list[0]

        return my_collate_fn_for_batch_size_being_1
    else:
        def my_collate_fn_for_batch_size_larger_than_1(batch_dict_list):
            return_dict = {key: list() for key in batch_dict_list[0].keys()}
            for item_dict in batch_dict_list:
                for k, v in item_dict.items():
                    return_dict[k].append(v)
            return return_dict

        return my_collate_fn_for_batch_size_larger_than_1


def init_a_figure_and_an_axes():
    """
    处理figure上的axes。清空axes上的内容，关闭axis，位置设为占满figure。
    如果还没有axes，则创建一个满足要求的axes。
    """
    figure = plt.figure(
        num="my_figure",
        frameon=False,
        dpi=DPI,
        clear=True,
    )
    if not figure.get_axes():
        axes = figure.add_axes((0, 0, 1, 1))
    else:
        axes = figure.axes[0]
        axes.set_position((0, 0, 1, 1))
    axes.clear()
    axes.axis("off")
    return figure, axes


def adjust_figure_size_and_show_image_and_release_resources(
        img,
        figure: plt.Figure,
        axes: plt.Axes
):
    """
    根据face尺寸调整figure尺寸。axes会自动随着figure调整位置，我们不用动axes。
    Args:
        img: HxWxC RGB
        figure:
        axes:
    Returns:
    """
    h, w, _ = img.shape  # HxWxC
    figure.set_size_inches(
        h=h / DPI,
        w=w / DPI,
    )

    axes.imshow(img)
    figure.show()

    plt.close(fig=figure)
    axes.clear()
    axes.axis("off")

    return figure


def save_to_json(obj, save_to_path):
    with open(save_to_path, "w") as file:
        json.dump(obj, file)


def load_json(load_from_path):
    with open(load_from_path, "r") as file:
        return json.load(file)


def build_dataset_for_test():
    from a004_main.a002_batch_test.a002_DatasetForTestOrVali import DatasetForTestOrVali
    return DatasetForTestOrVali(
        dataset_path=DATASET_SF_TL54_PATH,
        num_samples_per_epoch=TEST_NUM_SAMPLES_PER_EPOCH
    )


def get_time_str():
    return datetime.now().strftime("%m-%d-%H-%M")


def loss_penalty_func_for_d_an(x):
    return math.sin(math.pi / 4 * x + math.pi) + 1


def my_distance_func(tensor_0, tensor_1) -> torch.Tensor:
    """
    假设tensor的形状为 batch x feature_dim，返回tensor形状将是只有一维，长度为batch
    """
    return 1 - functional.cosine_similarity(tensor_0, tensor_1)
