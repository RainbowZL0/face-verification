import base64
from io import BytesIO
from PIL import Image
import numpy as np

base64_txt_path = "./a004_main/a005_grammer_test/base64_png.txt"


def test_01():
    with open(base64_txt_path, 'r') as file:
        base64_str = file.read()

        # 如果有头部信息，先去掉
        if "base64," in base64_str:
            base64_str = base64_str.split(",")[1]

        # 解码Base64字符串为二进制数据
        image_data = base64.b64decode(base64_str)

        # 将二进制数据转换为Pillow图像对象
        image = Image.open(BytesIO(image_data))
        # 转换为numpy数组
        image_array = np.array(image)

        # 输出结果
        print("图片的ndarray形状：", image_array.shape)


if __name__ == '__main__':
    test_01()
    pass
