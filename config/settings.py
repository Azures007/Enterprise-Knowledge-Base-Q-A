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
    """检索后最终返回的最相关文档块数量"""

    RETRIEVAL_CANDIDATE_K: int = 20
    """向量检索的候选块数量（重排前的粗召回，重排后取前 RETRIEVAL_TOP_K）"""

    HYBRID_SEARCH_ENABLED: bool = True
    """是否启用混合检索（向量 0.7 + 关键词 0.3 融合）。提升专有名词/编号类查询召回"""

    QUERY_REWRITE_ENABLED: bool = True
    """是否启用多轮问题重写（结合历史把当前问题改写成独立完整问题）"""

    RELATED_QUESTIONS_ENABLED: bool = True
    """是否在回答后生成相关问题推荐"""

    CITE_SOURCES_ENABLED: bool = True
    """是否启用回答内嵌 [N] 引用标注"""

    RERANK_ENABLED: bool = True
    """是否启用重排。优先使用本地交叉编码器（bge-reranker），
       依赖不可用时自动降级为关键词重合的轻量重排"""

    RERANKER_TIMEOUT: int = 30
    """交叉编码器模型加载超时（秒）。模型需从 Hugging Face 下载时，
       超过该时间自动降级为轻量重排，避免阻塞应用启动"""

    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    """交叉编码器模型名（可选，安装 sentence-transformers 后生效）"""

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
    # 认证配置（JWT + API Key）
    # ================================================================
    AUTH_ENABLED: bool = True
    """是否启用 API 认证（默认开启；仅 /api/auth/login 与 /api/health 免认证）"""

    JWT_SECRET_KEY: str = "change-me-in-production-please-use-a-random-32-byte-key"
    """JWT 签名密钥（生产环境务必通过 .env 设置为随机字符串，建议 ≥32 字节）"""

    JWT_ALGORITHM: str = "HS256"
    """JWT 签名算法"""

    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7
    """JWT 令牌有效期（分钟），默认 7 天"""

    AUTH_ADMIN_USERNAME: str = "admin"
    """管理员用户名（首次登录时创建）"""

    AUTH_ADMIN_PASSWORD: str = "admin123"
    """管理员密码（首次登录时创建，生产环境务必修改）"""

    API_KEYS: str = ""
    """静态 API Key 列表，逗号分隔（如: key1,key2），非空时可用 X-API-Key 头认证"""

    # ================================================================
    # 缓存配置
    # ================================================================
    CACHE_MAX_ENTRIES: int = 10000
    """内存缓存最大条目数，超出后按 LRU 淘汰最久未使用的条目"""

    CACHE_QA_TTL: int = 3600
    """问答缓存 TTL（秒）"""

    # ================================================================
    # 查询审计配置
    # ================================================================
    AUDIT_ENABLED: bool = True
    """是否启用查询审计（记录每次问答到 query_audit_log）"""

    # ================================================================
    # 工具调用配置（Function Calling / Agentic 检索）
    # ================================================================
    TOOL_CALLING_ENABLED: bool = False
    """是否启用工具调用模式（LLM 自主决定调用哪些工具检索知识库）。默认关闭，渐进启用"""

    TOOL_CALLING_MAX_ROUNDS: int = 4
    """工具循环最大轮数，防止 LLM 反复调工具死循环"""

    TOOL_CALLING_RESULT_LIMIT: int = 2000
    """工具结果截断字符数，防止上下文爆炸"""

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
