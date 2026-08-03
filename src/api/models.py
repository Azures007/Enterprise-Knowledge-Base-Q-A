"""
=============================================================================
API 请求/响应 Pydantic 模型

为所有 API 端点提供类型安全的请求体验证和结构化响应，使 Swagger 文档
能够自动展示完整的 Schema 定义。
=============================================================================
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ==============================================================================
# 知识库问答
# ==============================================================================

class QueryRequest(BaseModel):
    """知识库问答请求体"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    k: Optional[int] = Field(None, ge=1, le=100, description="检索文档块数量")
    concise: bool = Field(False, description="是否使用简洁回答模式")
    filter: Optional[dict[str, Any]] = Field(None, description="元数据过滤条件")
    collection: Optional[str] = Field(None, description="指定查询的集合名称，不传则自动路由")


class StreamQueryRequest(BaseModel):
    """流式知识库问答请求体（SSE）"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    k: Optional[int] = Field(None, ge=1, le=100, description="检索文档块数量")
    concise: bool = Field(False, description="是否使用简洁回答模式")
    collection: Optional[str] = Field(None, description="指定查询的集合名称，不传则自动路由")


class SourceInfo(BaseModel):
    """回答来源信息"""

    filename: str = Field(..., description="文件名")
    source: str = Field("", description="来源路径")
    page: Optional[int] = Field(None, description="PDF 页码")
    slide: Optional[int] = Field(None, description="PPT 幻灯片页码")
    score: float = Field(0.0, description="相似度得分")


class QueryStats(BaseModel):
    """问答统计信息"""

    retrieved_chunks: int = Field(0, description="检索到的文档块数量")
    unique_sources: int = Field(0, description="去重后的来源数量")


class QueryResponseData(BaseModel):
    """知识库问答响应数据"""

    question: str
    answer: str
    sources: list[SourceInfo] = []
    answer_type: str = "general"
    collection: Optional[str] = None
    stats: QueryStats = QueryStats()


class APIResponse(BaseModel):
    """统一 API 响应包装"""

    code: int = Field(0, description="业务状态码，0 表示成功")
    message: str = Field("success", description="响应消息")
    data: Any = Field(None, description="响应数据")


# ==============================================================================
# 文档导入
# ==============================================================================

class IngestConfirmRequest(BaseModel):
    """OSS 直传确认请求体"""

    object_key: str = Field(..., min_length=1, description="OSS 对象键")
    filename: Optional[str] = Field(None, description="原始文件名")
    collection: Optional[str] = Field(None, description="目标集合名称")


# ==============================================================================
# 知识库集合管理
# ==============================================================================

class CreateCollectionRequest(BaseModel):
    """创建集合请求体"""

    name: str = Field(..., min_length=1, max_length=255, description="集合名称")


class RenameCollectionRequest(BaseModel):
    """重命名集合请求体"""

    name: str = Field(..., min_length=1, max_length=255, description="新集合名称")


# ==============================================================================
# 对话管理
# ==============================================================================

class BatchDeleteConversationsRequest(BaseModel):
    """批量删除对话请求体"""

    ids: list[int] = Field(..., min_length=1, description="要删除的对话 ID 列表")


class AddMessageRequest(BaseModel):
    """添加消息请求体"""

    role: str = Field("user", pattern="^(user|ai|system)$", description="消息角色")
    content: str = Field(..., min_length=1, description="消息内容")
    sources: Optional[list[dict[str, Any]]] = Field(None, description="引用来源")
    answer_type: Optional[str] = Field(None, description="回答类型")


class UpdateMessageRequest(BaseModel):
    """更新消息请求体"""

    content: str = Field(..., min_length=1, description="更新后的消息内容")
    sources: Optional[list[dict[str, Any]]] = Field(None, description="引用来源")


class UpdateConversationTitleRequest(BaseModel):
    """修改对话标题请求体"""

    title: str = Field(..., min_length=1, max_length=255, description="对话标题")
