"""
=============================================================================
用户集合隔离工具

方案 A（升级版）：集合级文档隔离 + 全部用户可创建集合。
- 每个用户拥有独立的个人集合，命名规则 `{username}的知识库`
- 用户可创建自己的集合，集合归属创建者（owner_id）
- 上传的文档自动归入用户的个人集合（或用户自建集合）
- 检索、列表、删除时按当前用户限定集合
- 管理员（is_admin=true）可见并操作所有集合

本模块只做"集合名 ↔ 用户"的映射与可见集合判断，不直接操作数据库。
=============================================================================
"""

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 管理员个人集合名称（兼容旧数据）
ADMIN_COLLECTION = "知识库"


def user_personal_collection(username: str) -> str:
    """返回用户的个人集合名称。"""
    if not username or username == "anonymous":
        return ADMIN_COLLECTION
    return f"{username}的知识库"


def is_admin_user(auth: dict) -> bool:
    """判断认证信息是否为管理员。"""
    return bool(auth.get("is_admin", False))


def visible_collections(
    auth: dict,
    all_collections: list[str],
    user_id: int | None = None,
    owner_map: dict[str, int | None] | None = None,
) -> list[str]:
    """
    根据用户身份过滤可见集合。

    - 管理员：可见全部集合
    - 普通用户：可见自己的个人集合 + 自己创建的集合（owner_id == user_id）
    - 匿名（认证关闭）：仅可见管理员集合（知识库）

    Args:
        auth:             require_auth 返回的认证信息
        all_collections:  全部集合名列表
        user_id:          当前用户的数据库 ID（用于匹配 owner_id）
        owner_map:        集合名 → owner_id 的映射（供判断归属）

    Returns:
        该用户可见的集合名列表
    """
    if is_admin_user(auth):
        return all_collections

    name = (auth.get("username") or "anonymous")
    own = user_personal_collection(name)

    if user_id is not None and owner_map is not None:
        # 普通用户：个人集合 + 自己创建的集合
        return [c for c in all_collections if c == own or owner_map.get(c) == user_id]

    # 无归属信息时退化为仅个人集合
    return [c for c in all_collections if c == own]
