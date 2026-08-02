"""
=============================================================================
OCR 文字识别模块

用于扫描版 PDF 和图片文件的文字识别。支持多引擎：
    1. PaddleOCR（首选，中文效果最好）
    2. EasyOCR（备选）
    3. 纯报错提示（都不可用时）

使用方法:
    from src.document_loader.ocr import ocr_image

    text = ocr_image("scan.jpg")
    text = ocr_image("scan.png", lang="ch")
=============================================================================
"""

import os
from pathlib import Path
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 尝试导入 OCR 引擎
_ocr_engine = None
_ocr_engine_name = None


def _get_engine():
    """惰性加载 OCR 引擎"""
    global _ocr_engine, _ocr_engine_name

    if _ocr_engine is not None:
        return _ocr_engine, _ocr_engine_name

    # 优先尝试 PaddleOCR
    try:
        from paddleocr import PaddleOCR

        # PaddleOCR 初始化时需要下载模型，静默下载
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
            use_gpu=False,
        )
        _ocr_engine_name = "paddleocr"
        logger.info("OCR 引擎: PaddleOCR（中文模式）")
        return _ocr_engine, _ocr_engine_name
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"PaddleOCR 初始化失败: {e}")

    # 备选 EasyOCR
    try:
        import easyocr

        _ocr_engine = easyocr.Reader(["ch_sim", "en"], gpu=False)
        _ocr_engine_name = "easyocr"
        logger.info("OCR 引擎: EasyOCR（中文模式）")
        return _ocr_engine, _ocr_engine_name
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"EasyOCR 初始化失败: {e}")

    # 都没有
    _ocr_engine_name = None
    return None, None


def ocr_image(image_path: str | Path, lang: str = "ch") -> str:
    """
    识别图片中的文字。

    Args:
        image_path: 图片文件路径
        lang:       语言 ("ch"=中文, "en"=英文)

    Returns:
        识别出的文本

    Raises:
        ImportError: 没有可用的 OCR 引擎
    """
    engine, engine_name = _get_engine()

    if engine is None:
        raise ImportError(
            "未安装 OCR 引擎。请安装 PaddleOCR 或 EasyOCR：\n"
            "  pip install paddlepaddle paddleocr   # PaddleOCR（推荐，中文效果最好）\n"
            "  或\n"
            "  pip install easyocr                    # EasyOCR（轻量备选）"
        )

    logger.info(f"正在 OCR 识别: {Path(image_path).name}（引擎: {engine_name}）")

    try:
        if engine_name == "paddleocr":
            result = engine.ocr(str(image_path), cls=True)
            return _parse_paddleocr(result)

        elif engine_name == "easyocr":
            result = engine.readtext(str(image_path))
            return _parse_easyocr(result)

    except Exception as e:
        logger.error(f"OCR 识别失败 [{Path(image_path).name}]: {e}")
        raise


def _parse_paddleocr(result: list) -> str:
    """解析 PaddleOCR 返回结果"""
    lines = []
    if not result or not result[0]:
        return ""

    for line in result[0]:
        text = line[1][0]  # (bbox, (text, confidence))
        if text and text.strip():
            lines.append(text.strip())

    return "\n".join(lines)


def _parse_easyocr(result: list) -> str:
    """解析 EasyOCR 返回结果"""
    lines = []
    if not result:
        return ""

    for bbox, text, confidence in result:
        if text and text.strip():
            lines.append(text.strip())

    return "\n".join(lines)


def ocr_pdf_page(page_image) -> str:
    """
    对 PDF 页面图片进行 OCR 识别。

    Args:
        page_image: PyMuPDF 页面渲染的 pixmap 对象

    Returns:
        识别出的文本
    """
    import tempfile

    # 保存为临时图片
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        page_image.save(tmp_path)
        return ocr_image(tmp_path)
    finally:
        os.unlink(tmp_path)
