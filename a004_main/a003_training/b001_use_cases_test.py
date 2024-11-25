import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from rich.pretty import pprint

from a004_main.a001_utils.a002_general_utils import loss_penalty_func_for_d_an


def test_cosine_similarity():
    a = torch.randn(20, 50)
    b = torch.randn(20, 50)
    c = 1 - torch.nn.functional.cosine_similarity(a, b)
    print(c.shape)  # Output: torch.Size([20])


def test_dataset_for_training_obj():
    from a004_main.a001_utils.a000_CONFIG import (
        DATASET_SF_TL54_PATH,
        DATASET_SF_TL54_CROPPED_PATH,
        TRAINING_DETECTOR_NAME, )
    from a004_main.a003_training.a002_DatasetForTraining import DatasetForTrainingAndVali

    dataset_for_training = DatasetForTrainingAndVali(
        original_dataset_path=DATASET_SF_TL54_PATH,
        num_samples_per_epoch=1000,
        create_or_exist_cropped_dataset_at_path=DATASET_SF_TL54_CROPPED_PATH,
        training_detector_name=TRAINING_DETECTOR_NAME,
        probability_for_mod_choices_for_training_dict=None,
        whether_build_cropped_dataset=False,
    )
    for i in range(5):
        pprint(dataset_for_training[i])


def test_my_training_obj():
    from a004_main.a001_utils.a000_CONFIG import (
        DATASET_SF_TL54_PATH,
        DATASET_SF_TL54_CROPPED_PATH,
        TRAINING_DETECTOR_NAME, TRAINING_NUM_SAMPLES_PER_EPOCH,
    )
    from a004_main.a003_training.a002_DatasetForTraining import DatasetForTrainingAndVali
    dataset_for_training = DatasetForTrainingAndVali(
        original_dataset_path=DATASET_SF_TL54_PATH,
        num_samples_per_epoch=TRAINING_NUM_SAMPLES_PER_EPOCH,
        create_or_exist_cropped_dataset_at_path=DATASET_SF_TL54_CROPPED_PATH,
        training_detector_name=TRAINING_DETECTOR_NAME,
        probability_for_mod_choices_for_training_dict=None,
        whether_build_cropped_dataset=False,
    )

    from a004_main.a003_training.a003_MyTrainingObj import (MyTrainingObj)
    my_training_obj = MyTrainingObj(dataset_for_training)
    my_training_obj.start_train_and_vali()


def handle_figure_and_2_axes():
    fig = plt.figure(dpi=300, layout='compressed', figsize=(20, 5))
    axes_array = fig.subplots(1, 2, sharex=True, sharey=True)

    gnr = np.random.default_rng()
    data_dict = {
        'data_1': gnr.normal(0, 1, 10000),
        'data_2': gnr.normal(0, 1, 10000)
    }

    sns.set_theme(style="darkgrid")
    sns.despine(fig)

    for index, (key, data) in enumerate(data_dict.items()):
        sns.histplot(
            data=data,
            kde=True,
            ax=axes_array[index],
            bins=200,
            stat='density',
            element='step',
        )
        axes_array[index].set_title(key)
        axes_array[index].set_xlabel('cosine_similarity')
        axes_array[index].set_xlim(-1, 1)

    fig.show()
    plt.close(fig)


def test_norm_func():
    x = np.linspace(0, 2, 1000)
    vec_norm_func = np.vectorize(loss_penalty_func_for_d_an)
    y = vec_norm_func(x)
    data = pd.DataFrame({'x': x, 'f(x)': y})

    fig = plt.figure(dpi=300, layout='compressed')
    sns.despine(fig)
    sns.set_style('darkgrid')
    axes = fig.subplots(1, 1)
    sns.lineplot(x='x', y='f(x)', data=data, ax=axes)
    fig.show()
    plt.close(fig)


if __name__ == '__main__':
    pass
    # test_dataset_for_training_obj()
    # test_cosine_similarity()
    # test_my_training_obj()
    # handle_figure_and_2_axes()
    test_norm_func()
