"""
=============================================================================
日志配置模块

提供统一的日志记录功能，支持同时输出到控制台和日志文件。
=============================================================================

使用方法:
    from src.utils.logger import setup_logger

    logger = setup_logger(__name__)
    logger.info("信息消息")
    logger.error("错误消息")
"""

import logging
import sys
from pathlib import Path

from config.settings import settings


def setup_logger(
    name: str,
    log_file: str | Path | None = None,
    level: str | None = None,
    console: bool = True,
) -> logging.Logger:
    """
    创建并配置一个日志记录器。

    Args:
        name:            日志记录器名称，通常传入 __name__
        log_file:        日志文件路径，None 则使用全局配置
        level:           日志级别，None 则使用全局配置
        console:         是否同时输出到控制台

    Returns:
        配置完成的 logging.Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复配置
    if logger.handlers:
        return logger

    # 日志级别
    log_level = getattr(logging, (level or settings.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(log_level)

    # --- 日志格式 ---
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- 文件处理器 ---
    log_path = Path(log_file or settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- 控制台处理器 ---
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
