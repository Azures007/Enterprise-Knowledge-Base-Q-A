"""
=============================================================================
文件存储抽象层

支持将原始文件存储到不同后端：
    - local: 本地磁盘（默认）
    - oss:  阿里云 OSS

使用方法:
    from src.storage import get_storage

    storage = get_storage()
    url = storage.save("file.pdf", file_bytes)
    storage.delete("file.pdf")
=============================================================================
"""

import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class StorageBackend(ABC):
    """存储后端抽象基类"""

    @abstractmethod
    def save(self, filename: str, data: bytes, content_type: str = "") -> str:
        """
        保存文件，返回访问路径/URL。

        Args:
            filename:    原始文件名
            data:        文件二进制内容
            content_type: MIME 类型

        Returns:
            文件的存储路径或 URL
        """
        ...

    @abstractmethod
    def delete(self, path: str) -> bool:
        """删除文件"""
        ...

    @abstractmethod
    def get_url(self, path: str) -> str:
        """获取文件的访问 URL"""
        ...


class LocalStorage(StorageBackend):
    """本地磁盘存储"""

    def __init__(self):
        self.base_dir = settings.PROJECT_ROOT / "data" / "files"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"本地存储初始化: {self.base_dir}")

    def save(self, filename: str, data: bytes, content_type: str = "") -> str:
        # 生成唯一文件名，避免冲突
        ext = Path(filename).suffix
        unique_name = f"{uuid.uuid4().hex}{ext}"
        file_path = self.base_dir / unique_name
        file_path.write_bytes(data)
        logger.info(f"文件已保存到本地: {file_path.name} ({len(data)} bytes)")
        return str(file_path)

    def delete(self, path: str) -> bool:
        p = Path(path)
        if p.exists():
            p.unlink()
            logger.info(f"本地文件已删除: {p.name}")
            return True
        return False

    def get_url(self, path: str) -> str:
        return path


class OSSStorage(StorageBackend):
    """阿里云 OSS 存储"""

    def __init__(self):
        self.endpoint = settings.OSS_ENDPOINT
        self.bucket_name = settings.OSS_BUCKET
        self.access_key = settings.OSS_ACCESS_KEY
        self.secret_key = settings.OSS_SECRET_KEY
        self.upload_dir = settings.OSS_UPLOAD_DIR

        if not self.endpoint or not self.bucket_name:
            raise ValueError(
                "OSS 配置不完整，请在 .env 中设置 OSS_ENDPOINT、OSS_BUCKET、"
                "OSS_ACCESS_KEY、OSS_SECRET_KEY"
            )

        try:
            import oss2
        except ImportError:
            raise ImportError(
                "需要安装 oss2 才能使用阿里云 OSS：pip install oss2"
            )

        auth = oss2.Auth(self.access_key, self.secret_key)
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        logger.info(f"OSS 存储初始化: bucket={self.bucket_name}, endpoint={self.endpoint}")

    def save(self, filename: str, data: bytes, content_type: str = "") -> str:
        ext = Path(filename).suffix
        object_key = f"{self.upload_dir}/{uuid.uuid4().hex}{ext}"

        headers = {}
        if content_type:
            headers["Content-Type"] = content_type

        try:
            result = self.bucket.put_object(object_key, data, headers=headers)
            if result.status != 200:
                raise RuntimeError(f"OSS 上传失败: status={result.status}")
            logger.info(f"文件已上传到 OSS: {object_key} ({len(data)} bytes)")
            return object_key
        except Exception as e:
            logger.error(f"OSS 上传失败: {e}")
            raise

    def delete(self, path: str) -> bool:
        try:
            self.bucket.delete_object(path)
            logger.info(f"OSS 文件已删除: {path}")
            return True
        except Exception as e:
            logger.warning(f"OSS 删除失败: {e}")
            return False

    def get_url(self, path: str) -> str:
        # 生成公共可访问的 URL
        return f"https://{self.bucket_name}.{self.endpoint.replace('https://', '')}/{path}"

    def generate_upload_url(self, filename: str, expires_in: int = 3600) -> dict:
        """
        生成预签名上传 URL（前端直传 OSS 用）。

        Args:
            filename:   原始文件名
            expires_in: URL 过期时间（秒）

        Returns:
            dict: {upload_url, object_key, filename, expires_in}
        """
        ext = Path(filename).suffix
        object_key = f"{self.upload_dir}/{uuid.uuid4().hex}{ext}"

        upload_url = self.bucket.sign_url(
            'PUT', object_key, expires_in,
            headers={'Content-Type': 'application/octet-stream'},
        )

        return {
            "upload_url": upload_url,
            "object_key": object_key,
            "filename": filename,
            "expires_in": expires_in,
        }

    def get_object(self, object_key: str) -> bytes:
        """从 OSS 下载文件内容"""
        import oss2
        try:
            result = self.bucket.get_object(object_key)
            return result.read()
        except oss2.exceptions.NoSuchKey as e:
            raise FileNotFoundError(f"OSS 文件不存在: {object_key}") from e


def get_storage() -> StorageBackend:
    """根据配置获取存储后端实例"""
    backend = settings.STORAGE_BACKEND
    if backend == "oss":
        return OSSStorage()
    return LocalStorage()