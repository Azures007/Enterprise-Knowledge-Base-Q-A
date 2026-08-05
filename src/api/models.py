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
    conversation_id: Optional[int] = Field(None, description="对话 ID，传入后使用该对话的历史进行多轮问答")


class StreamQueryRequest(BaseModel):
    """流式知识库问答请求体（SSE）"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    k: Optional[int] = Field(None, ge=1, le=100, description="检索文档块数量")
    concise: bool = Field(False, description="是否使用简洁回答模式")
    collection: Optional[str] = Field(None, description="指定查询的集合名称，不传则自动路由")
    conversation_id: Optional[int] = Field(None, description="对话 ID，传入后使用该对话的历史进行多轮问答")


class SourceChunk(BaseModel):
    """来源文档中被检索到的分块"""

    content: str = Field("", description="分块内容")
    score: float = Field(0.0, description="该块的相似度分数")
    chunk_index: Optional[int] = Field(None, description="分块在文档内的序号")


class SourceInfo(BaseModel):
    """回答来源信息"""

    index: Optional[int] = Field(None, description="引用编号，对应回答中的 [N] 标注")
    filename: str = Field(..., description="文件名")
    source: str = Field("", description="来源路径")
    page: Optional[int] = Field(None, description="PDF 页码")
    slide: Optional[int] = Field(None, description="PPT 幻灯片页码")
    score: float = Field(0.0, description="相似度得分")
    chunks: list[SourceChunk] = Field(default_factory=list, description="该文档中被检索到的分块列表")


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
    related_questions: list[str] = Field(default_factory=list, description="推荐的相关问题")


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


class MessageFeedbackRequest(BaseModel):
    """消息反馈请求体"""

    feedback: int = Field(..., description="1=赞, -1=踩, 0=清除")
    comment: Optional[str] = Field(None, max_length=500, description="点踩原因（可选）")


class UpdateConversationTitleRequest(BaseModel):
    """修改对话标题请求体"""

    title: str = Field(..., min_length=1, max_length=255, description="对话标题")


# ==============================================================================
# 认证
# ==============================================================================

class LoginRequest(BaseModel):
    """登录请求体"""

    username: str = Field(..., min_length=1, max_length=255, description="用户名")
    password: str = Field(..., min_length=1, max_length=255, description="密码")


class LoginResponseData(BaseModel):
    """登录响应数据"""

    token: str = Field(..., description="JWT 令牌")
    token_type: str = Field("bearer", description="令牌类型")
    expires_in: int = Field(..., description="令牌有效期（秒）")
    username: str = Field(..., description="用户名")
    is_admin: bool = Field(False, description="是否为管理员")


# ==============================================================================
# 用户管理（仅管理员）
# ==============================================================================

class CreateUserRequest(BaseModel):
    """创建用户请求体"""

    username: str = Field(..., min_length=1, max_length=255, description="用户名（唯一）")
    password: str = Field(..., min_length=6, max_length=255, description="密码（至少 6 位）")
    display_name: Optional[str] = Field(None, max_length=255, description="显示名")
    is_admin: bool = Field(False, description="是否为管理员")


class ChangePasswordRequest(BaseModel):
    """用户自助修改密码请求体"""

    old_password: str = Field(..., min_length=1, max_length=255, description="原密码")
    new_password: str = Field(..., min_length=6, max_length=255, description="新密码（至少 6 位）")


class ResetPasswordRequest(BaseModel):
    """管理员重置用户密码请求体"""

    new_password: str = Field(..., min_length=6, max_length=255, description="新密码（至少 6 位）")
