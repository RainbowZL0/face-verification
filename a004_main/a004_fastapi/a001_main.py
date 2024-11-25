from colorama import Fore
from fastapi import FastAPI, UploadFile, File
from uvicorn import run

from a004_main.a004_fastapi.a002_my_fastapi_processor import MyFastapiProcessor

app = FastAPI()
my_fastapi_processor = MyFastapiProcessor()


@app.get("/get_data")
def get_data():
    return {"message": "This is a GET response"}


@app.post("/send_data")
def send_data(data):
    return {"received": data}


@app.post("/upload_image_pair_and_verify")
def upload_image_pair_and_verify(
        file_0: UploadFile = File(...),
        file_1: UploadFile = File(...),
):
    """
    1. 类型注解UploadFile被称为'请求体参数类型'，发来的数据将被解析为该类的对象。可以使用自定义的类，只要继承BaseModel类。
    2. 看上去像参数默认值的File(...)被称为'请求参数声明函数'，虽然首字母大写，不过不是类，而是方法。专门用于校验传入参数的格式。
        如果满足要求，才会被进一步封装为UploadFile类的对象。
    3. 类型注解UploadFile在这里不只是注释，还会影响运行功能。
    """
    return my_fastapi_processor.get_image_pair_and_verify_file_version(file_0=file_0, file_1=file_1)


def start():
    from a004_main.a001_utils.a000_CONFIG import LOGGER
    LOGGER.info(
        Fore.GREEN +
        "服务即将启动，调试可使用Swagger http://127.0.0.1:8000/docs 或 Redoc http://127.0.0.1:8000/redoc。"
    )
    run(app='a001_main:app', reload=False)


if __name__ == '__main__':
    start()
