import seaborn as sns
from matplotlib import pyplot as plt


def test_histogram():
    figure, axes = plt.subplots(nrows=1, ncols=1)
    figure.set_layout_engine("constrained")

    sns.set_theme(style="ticks")  # 设置一个主题样式
    sns.despine(figure)  # 去掉顶部和右侧框线

    data = sns.load_dataset("diamonds")
    sns.histplot(
        data=data,
        x="price",
        # kde=True,
        ax=axes,
        # hue="cut",
        multiple="stack",
        # palette="light:m_r",
        # log_scale=True,
        # linewidth=.5,
        bins=1000,
    )
    figure.show()
    plt.close(fig=figure)


if __name__ == "__main__":
    pass
    test_histogram()
