import logging
from colorama import Fore, init
init(autoreset=True)


def build_logger_with_handler():
    # 创建 Logger
    logger = logging.getLogger(name="my_logger")
    logger.setLevel(logging.DEBUG)

    # logger可以关联多个handler，用于输出到不同位置，例如控制台或文件
    console_handler = logging.StreamHandler()
    # 设置handler的过滤级别
    console_handler.setLevel(logging.DEBUG)
    # 创建 Formatter并设置格式
    console_handler.setFormatter(
        logging.Formatter(
            Fore.WHITE +
            '----------------------------------------------------------------\n'
            '自定义运行时调试信息: File "%(pathname)s", line %(lineno)d, %(levelname)s, \n'
            '%(message)s\n' +
            Fore.WHITE +
            '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        )
    )

    # 将 Handler 添加到 Logger
    logger.addHandler(console_handler)

    # 输出日志
    # logger.debug(Fore.LIGHTMAGENTA_EX + '创建了一个logger，用于输出可跳转到代码行的日志。')
    # logger.info('This is an info message')
    # logger.warning('This is a warning message')
    # logger.error('This is an error message')
    # logger.critical('This is a critical message')
    return logger
