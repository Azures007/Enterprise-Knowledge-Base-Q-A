"""
=============================================================================
用户管理模块

基于数据库 users 表提供用户认证与密码管理：
    - PBKDF2-SHA256 密码哈希（带随机盐，防彩虹表）
    - admin 账号的初始化（首次启动时自动创建）

使用方法:
    from src.users import UserManager, hash_password, verify_password

    um = UserManager(vector_store)
    await um.ensure_admin()          # 确保 admin 存在
    user = await um.authenticate("admin", "admin123")
=============================================================================
"""

import hashlib
import hmac
import secrets

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# PBKDF2 迭代次数（OWASP 建议 ≥600k；开发环境用 100k 平衡速度）
_PBKDF2_ITERATIONS = 100_000


def hash_password(password: str) -> str:
    """
    使用 PBKDF2-SHA256 + 随机盐生成密码哈希。

    格式: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    校验密码是否与存储的哈希匹配。

    Args:
        password:    明文密码
        stored_hash: hash_password() 生成的哈希

    Returns:
        是否匹配
    """
    try:
        algorithm, iterations_str, salt_hex, hash_hex = stored_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)

        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        # 常量时间比较，防时序攻击
        return hmac.compare_digest(digest, expected)
    except (ValueError, TypeError):
        return False


class UserManager:
    """
    用户管理器：负责用户查询、认证、admin 初始化。

    与 PGVectorStore 解耦：只依赖 store 提供的用户异步方法，
    因此也能与异步/同步两种调用方式配合。
    """

    def __init__(self, vector_store=None):
        """
        Args:
            vector_store: 实现用户方法的存储实例（默认自动创建 PGVectorStore）
        """
        if vector_store is None:
            from src.embeddings import BailianEmbeddings
            from src.vector_store import PGVectorStore
            vector_store = PGVectorStore(BailianEmbeddings())
        self._store = vector_store

    async def ensure_admin(self):
        """
        确保 admin 账号存在。

        若 users 表中没有 admin，则根据配置（AUTH_ADMIN_USERNAME/PASSWORD）
        创建，密码哈希化后入库。
        """
        username = settings.AUTH_ADMIN_USERNAME
        password = settings.AUTH_ADMIN_PASSWORD

        existing = await self._store.aget_user_by_username(username)
        if existing is not None:
            logger.info(f"admin 用户已存在: {username}")
            return existing

        password_hash = hash_password(password)
        try:
            user_id = await self._store.acreate_user(
                username=username,
                password_hash=password_hash,
                display_name="Administrator",
                is_admin=True,
            )
            logger.warning(
                f"已创建初始 admin 账号: {username} "
                f"(密码来自 AUTH_ADMIN_PASSWORD 配置，生产环境请立即修改)"
            )
            return {"id": user_id, "username": username, "is_admin": True}
        except Exception as e:
            logger.warning(f"创建 admin 用户失败: {e}")
            return None

    async def authenticate(self, username: str, password: str) -> dict | None:
        """
        校验用户名密码。

        Returns:
            校验通过返回用户信息（不含密码哈希）；失败返回 None
        """
        user = await self._store.aget_user_by_username(username)
        if user is None:
            return None
        if not verify_password(password, user["password_hash"]):
            return None

        # 更新最近登录时间
        try:
            await self._store.aupdate_user_login(user["id"])
        except Exception:
            pass

        return {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "is_admin": user["is_admin"],
        }

    async def create_user(
        self,
        username: str,
        password: str,
        display_name: str | None = None,
        is_admin: bool = False,
    ) -> dict:
        """
        创建新用户。

        Args:
            username:    用户名（唯一）
            password:    明文密码（内部哈希后存储）
            display_name: 显示名（可选）
            is_admin:    是否为管理员

        Returns:
            新用户信息 dict

        Raises:
            ValueError: 用户名/密码不合法或已存在
        """
        username = (username or "").strip()
        if not username:
            raise ValueError("用户名不能为空")
        if len(username) > 255:
            raise ValueError("用户名过长（最多 255 字符）")
        if not password or len(password) < 6:
            raise ValueError("密码至少 6 位")

        password_hash = hash_password(password)
        try:
            user_id = await self._store.acreate_user(
                username=username,
                password_hash=password_hash,
                display_name=display_name,
                is_admin=is_admin,
            )
        except Exception as e:
            if "已存在" in str(e):
                raise ValueError(f"用户名 '{username}' 已存在") from e
            raise

        logger.info(f"管理员创建用户: {username} (id={user_id}, is_admin={is_admin})")
        return {
            "id": user_id,
            "username": username,
            "display_name": display_name,
            "is_admin": is_admin,
        }

    async def list_users(self) -> list[dict]:
        """列出所有用户（不含密码哈希）。"""
        return await self._store.alist_users()

    async def delete_user(self, user_id: int) -> bool:
        """删除用户（级联删除其对话）。"""
        return await self._store.adelete_user(user_id)

    async def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str,
    ) -> bool:
        """
        用户自助修改密码。

        Args:
            username:     用户名
            old_password: 原密码（需验证正确）
            new_password: 新密码

        Returns:
            True 修改成功；False 原密码错误

        Raises:
            ValueError: 新密码不合法（<6 位）
        """
        if not new_password or len(new_password) < 6:
            raise ValueError("新密码至少 6 位")

        # 校验原密码
        user = await self.authenticate(username, old_password)
        if user is None:
            return False

        new_hash = hash_password(new_password)
        ok = await self._store.aupdate_user_password(user["id"], new_hash)
        if ok:
            logger.info(f"用户 {username} 已修改密码")
        return ok

    async def reset_password(self, admin_username: str, target_user_id: int, new_password: str) -> bool:
        """
        管理员重置指定用户的密码。

        Args:
            admin_username: 操作的管理员用户名（用于日志）
            target_user_id: 目标用户 ID
            new_password:   新密码

        Returns:
            True 重置成功；False 用户不存在

        Raises:
            ValueError: 新密码不合法
        """
        if not new_password or len(new_password) < 6:
            raise ValueError("新密码至少 6 位")

        new_hash = hash_password(new_password)
        ok = await self._store.aupdate_user_password(target_user_id, new_hash)
        if ok:
            logger.info(f"管理员 {admin_username} 重置了用户 #{target_user_id} 的密码")
        return ok
