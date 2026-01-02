"""
日志模块 - STAR-Forecast
提供统一的日志记录功能
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import colorlog


def setup_logger(name='STAR-Forecast', log_level='INFO', log_file=None, console_output=True):
    """
    设置和配置日志记录器

    参数:
        name: 日志记录器名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则不保存到文件
        console_output: 是否在控制台输出

    返回:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 设置日志格式
    console_format = '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # 颜色配置
        color_formatter = colorlog.ColoredFormatter(
            console_format,
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 创建轮转文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        file_formatter = logging.Formatter(
            file_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name='STAR-Forecast'):
    """
    获取日志记录器

    参数:
        name: 日志记录器名称

    返回:
        logging.Logger: 日志记录器
    """
    return logging.getLogger(name)


class LoggerMixin:
    """提供日志功能的混入类"""

    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__
            self._logger = get_logger(class_name)
        return self._logger

    def log_debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def log_info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def log_warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def log_error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def log_critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)


# 创建默认的全局日志记录器
default_logger = setup_logger()