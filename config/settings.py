"""
=============================================================================
项目全局配置模块

从 .env 文件和系统环境变量中加载所有配置项，提供统一的配置访问入口。
使用 pydantic-settings 实现类型安全的配置管理。
=============================================================================

使用方法:
    from config.settings import settings
    print(settings.BAILIAN_API_KEY)      # 阿里云百炼 API 密钥
    print(settings.LLM_MODEL_NAME)       # 当前使用的 LLM 模型名
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    全局配置类，所有配置项集中管理。

    配置加载优先级（高 → 低）：
        1. 系统环境变量 (os.environ)
        2. .env 文件中的变量
        3. 此处定义的默认值
    """

    # ================================================================
    # 阿里云百炼 API 配置
    # ================================================================
    BAILIAN_API_KEY: str = ""  # 从 .env 加载
    """阿里云百炼 API 密钥"""

    BAILIAN_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    """API 基础地址，使用兼容 OpenAI 模式的地址"""

    # ================================================================
    # 服务配置
    # ================================================================
    HOST: str = "0.0.0.0"
    """服务监听地址"""

    PORT: int = 8000
    """服务监听端口"""

    DEFAULT_COLLECTION: str = "knowledge_base"
    """默认集合名称"""

    # ================================================================
    # 模型配置
    # ================================================================
    LLM_MODEL_NAME: str = "qwen-plus"
    """文本生成模型名称，可选: qwen-plus, qwen-max, qwen-turbo"""

    EMBEDDING_MODEL_NAME: str = "text-embedding-v3"
    """文本嵌入模型名称，可选: text-embedding-v3, text-embedding-v2"""

    # ================================================================
    # 向量数据库配置
    # ================================================================
    VECTOR_DB_PATH: str = "./data/chroma_db"
    """ChromaDB 持久化存储路径"""

    # ================================================================
    # 文档处理配置
    # ================================================================
    CHUNK_SIZE: int = 500
    """文本分块大小（按字符数切割）"""

    CHUNK_OVERLAP: int = 100
    """文本分块重叠大小，保持上下文连贯性"""

    # ================================================================
    # 检索配置
    # ================================================================
    RETRIEVAL_TOP_K: int = 5
    """检索时返回的最相关文档块数量"""

    # ================================================================
    # 数据库配置（PostgreSQL + pgvector）
    # ================================================================
    VECTOR_STORE_TYPE: str = "pg"
    """向量存储后端类型: pg（PostgreSQL+pgvector）| chroma（ChromaDB）"""

    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/knowledge_base"
    """PostgreSQL 连接字符串（用于替代 ChromaDB/SQLite）"""

    VECTOR_DIMENSION: int = 1024
    """向量维度（与 text-embedding-v3 一致）"""

    VECTOR_INDEX_TYPE: str = "hnsw"
    """向量索引类型: hnsw（高精度）| ivfflat（快速建索引）"""

    # ================================================================
    # 对象存储配置（原始文件）
    # ================================================================
    STORAGE_BACKEND: str = "local"
    """文件存储后端: local | oss | s3"""

    OSS_ENDPOINT: str = ""
    """OSS/S3 端点地址"""

    OSS_BUCKET: str = ""
    """OSS/S3 存储桶名称"""

    OSS_ACCESS_KEY: str = ""
    """OSS/S3 访问密钥"""

    OSS_SECRET_KEY: str = ""
    """OSS/S3 秘密密钥"""

    OSS_UPLOAD_DIR: str = "knowledge_base_files"
    """OSS/S3 上传目录前缀"""

    # ================================================================
    # 限流与 Redis 配置
    # ================================================================
    REDIS_HOST: str = "localhost"
    """Redis 主机地址"""

    REDIS_PORT: int = 6379
    """Redis 端口"""

    REDIS_DB: int = 0
    """Redis 数据库编号"""

    RATE_LIMIT_IP_QPS: int = 10
    """每 IP 每秒最大请求数"""

    RATE_LIMIT_GLOBAL_QPS: int = 50
    """全局每秒最大请求数"""

    RATE_LIMIT_LLM_QPS: int = 8
    """百炼 API 每秒最大请求数（预留安全余量）"""

    RATE_LIMIT_LLM_BURST: int = 16
    """百炼 API 令牌桶容量（突发峰值限制）"""

    # ================================================================
    # 日志配置
    # ================================================================
    LOG_LEVEL: str = "INFO"
    """日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL"""

    LOG_FILE: str = "./logs/app.log"
    """日志文件路径"""

    # ================================================================
    # 项目路径
    # ================================================================
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    """项目根目录路径"""

    @property
    def DOCUMENTS_DIR(self) -> Path:
        """文档存放目录"""
        return self.PROJECT_ROOT / "data" / "documents"

    @property
    def PROCESSED_DIR(self) -> Path:
        """处理后文本缓存目录"""
        return self.PROJECT_ROOT / "data" / "processed"

    # ================================================================
    # pydantic-settings 配置
    # ================================================================
    model_config = SettingsConfigDict(
        env_file=".env",          # 从 .env 文件加载
        env_file_encoding="utf-8",
        case_sensitive=True,      # 保持大小写敏感
        extra="ignore",           # 忽略多余的字段
    )


# ==============================================================================
# 全局单例：在项目任何地方通过 from config.settings import settings 导入
# ==============================================================================
settings = Settings()

# 确保必要目录存在
settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 启动校验：确保 API Key 已配置
# ==============================================================================
if not settings.BAILIAN_API_KEY:
    raise RuntimeError(
        'BAILIAN_API_KEY 未配置。请在 .env 文件中设置:\n'
        '  BAILIAN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx\n'
        '详见 .env.example 模板。'
    )
