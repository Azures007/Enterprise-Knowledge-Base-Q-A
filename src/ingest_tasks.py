"""
=============================================================================
异步文档导入任务管理器

大文件（>10MB）上传后，同步解析/分块/入库会长时间占用事件循环，
阻塞其他请求。本模块将大文件导入放到后台协程任务执行：

    - submit():  接受文件元数据，创建任务记录并立即返回 task_id
    - 后台任务:   在独立 asyncpg 连接 + 独立 RAG 管线中完成
                 解析 → 分块 → 嵌入 → 入库，全程更新任务进度
    - get_status(): 查询任务状态供前端轮询

设计要点：
    1. 使用 asyncio.create_task 进程内协程，无需引入 Celery/ARQ 等外部队列
    2. 后台任务使用独立的 RAG 管线实例（不复用全局单例），
       避免与在线问答请求竞争同一 asyncpg 连接池
    3. 文档记录 + 分块入库放在同一数据库事务中，失败整体回滚，
       从根源上杜绝"只有元数据、无分块"的僵尸记录
    4. 任务状态持久化在 ingest_audit_log 表（扩展了 task_id/progress 字段）
=============================================================================
"""

import asyncio
import hashlib
import tempfile
import traceback
import uuid
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 大文件阈值：超过此大小走异步导入（字节）
ASYNC_INGEST_THRESHOLD = 10 * 1024 * 1024  # 10MB

# 导入流程各阶段进度值（0~100）
PROGRESS_STAGES = {
    "downloaded": 10,
    "deduped": 20,
    "parsed": 50,
    "chunked": 70,
    "embedded": 90,
    "done": 100,
}


class IngestTaskManager:
    """
    异步导入任务管理器。

    以进程内单例方式挂载到 app.state.ingest_task_mgr。
    """

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    # ================================================================
    # 公开接口
    # ================================================================

    async def submit(
        self,
        filename: str,
        content: bytes,
        collection: str | None,
        rag,
        loader,
        chunker,
        store,
    ) -> dict:
        """
        提交异步导入任务。

        Args:
            filename:   上传文件名
            content:    文件二进制内容
            collection: 目标集合（可选）
            rag:        RAG 管线（仅用于读取配置，后台任务会新建独立实例）
            loader:     文档加载器
            chunker:    文本分块器
            store:      向量存储（用于记录任务状态）

        Returns:
            dict: 含 task_id, status, filename
        """
        task_id = uuid.uuid4().hex
        file_size = len(content)

        # 创建任务记录
        await store.acreate_ingest_task(
            task_id=task_id,
            filename=filename,
            collection=collection or "default",
            file_size=file_size,
        )

        # 后台执行（进程内协程任务）
        task = asyncio.create_task(
            self._run_import(
                task_id=task_id,
                filename=filename,
                content=content,
                collection=collection,
                loader=loader,
                chunker=chunker,
            )
        )
        self._tasks[task_id] = task
        # 任务结束清理
        task.add_done_callback(lambda t: self._tasks.pop(task_id, None))

        logger.info(f"异步导入任务已创建: {task_id} ({filename}, {file_size} bytes)")
        return {
            "task_id": task_id,
            "status": "pending",
            "filename": filename,
            "file_size": file_size,
        }

    async def get_status(self, task_id: str, store) -> dict | None:
        """查询任务状态。"""
        return await store.aget_ingest_task(task_id)

    async def list_active(self, store) -> list[dict]:
        """列出进行中的任务（pending / processing）。"""
        rows = await store.alist_active_ingest_tasks()
        return rows

    # ================================================================
    # 后台导入逻辑
    # ================================================================

    async def _run_import(
        self,
        task_id: str,
        filename: str,
        content: bytes,
        collection: str | None,
        loader,
        chunker,
    ):
        """
        后台执行完整导入流程。

        使用独立 RAG 管线与独立数据库连接，整个入库放同一事务中。
        """
        import json
        import os

        from config.settings import settings
        from src.embeddings import BailianEmbeddings
        from src.vector_store import PGVectorStore

        tmp_path = None
        try:
            # 独立向量存储（独立连接池，不干扰在线问答）
            embedder = BailianEmbeddings()
            store = PGVectorStore(embedder)
            store.collection_name = collection or settings.DEFAULT_COLLECTION

            file_ext = Path(filename).suffix.lower()

            # ---- 保存临时文件 ----
            tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
            tmp_path = tmp_file.name
            tmp_file.write(content)
            tmp_file.close()

            content_hash = hashlib.sha256(content).hexdigest()

            # 查重（内容相同）
            existing = await store.afind_document_by_hash(content_hash)
            if existing is not None:
                existing_chunks = await store.acount_document_chunks(existing["id"])
                if existing_chunks > 0:
                    # 真正的重复
                    await store.aupdate_ingest_task(
                        task_id, status="failed", progress=20,
                        error_msg=f"文件内容与 {existing['filename']} 重复",
                    )
                    return
                else:
                    # 僵尸记录清理
                    try:
                        await store.adelete_document(existing["id"], delete_storage=True)
                    except Exception:
                        pass

            # 文件名查重：版本化覆盖（旧版本保留记录/chunks，标记失效供回滚）
            existing_name = await store.afind_document_by_filename(filename)
            old_version = 1
            prev_doc_id = None
            if existing_name is not None:
                try:
                    await store.areplace_document_for_new_version(existing_name["id"])
                    old_version = await store._aget_document_version(existing_name["id"])
                    prev_doc_id = existing_name["id"]
                except Exception:
                    pass

            await store.aupdate_ingest_task(task_id, progress=PROGRESS_STAGES["deduped"])

            # ---- 解析 ----
            raw_docs = loader.load_file(tmp_path)
            if not raw_docs:
                await store.aupdate_ingest_task(
                    task_id, status="failed", progress=50,
                    error_msg="文档解析后未提取到任何文本",
                )
                return

            # 修正文件名
            for doc in raw_docs:
                if "metadata" in doc:
                    doc["metadata"]["filename"] = filename
                    doc["metadata"]["source"] = filename

            await store.aupdate_ingest_task(task_id, progress=PROGRESS_STAGES["parsed"])

            # ---- 分块 ----
            chunked_docs = chunker.split_documents(raw_docs)
            await store.aupdate_ingest_task(task_id, progress=PROGRESS_STAGES["chunked"])

            # ---- 嵌入 + 入库（同一事务） ----
            pool = await store._aconn()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    # 记录文件元数据
                    doc_id = await conn.fetchval(
                        """
                        INSERT INTO documents
                        (collection_id, filename, file_type, file_size, content_hash, storage_backend, storage_path)
                        VALUES ($1, $2, $3, $4, $5, 'local', $6)
                        RETURNING id
                        """,
                        await store._aensure_collection(store.collection_name, conn),
                        filename,
                        file_ext.lstrip("."),
                        len(content),
                        content_hash,
                        filename,
                    )

                    # 版本化：若存在旧版本，为新文档设置版本号与版本链
                    if prev_doc_id is not None:
                        await conn.execute(
                            "UPDATE documents SET version = $1, prev_doc_id = $2 WHERE id = $3",
                            old_version + 1, prev_doc_id, doc_id,
                        )

                    # 计算嵌入向量
                    texts = [d["page_content"] for d in chunked_docs]
                    metadatas = [d.get("metadata", {}) for d in chunked_docs]
                    if hasattr(embedder, "aembed_documents"):
                        embeddings = await embedder.aembed_documents(texts)
                    else:
                        embeddings = embedder.embed_documents(texts)

                    # 批量写入 chunks
                    rows = []
                    for j in range(len(texts)):
                        meta_json = json.dumps(metadatas[j], ensure_ascii=False)
                        emb_str = "[" + ",".join(str(v) for v in embeddings[j]) + "]"
                        rows.append((doc_id, texts[j], meta_json, emb_str))
                    await conn.executemany(
                        """
                        INSERT INTO chunks (collection_id, document_id, content, metadata, embedding)
                        VALUES ((SELECT collection_id FROM documents WHERE id = $1), $1, $2, $3::jsonb, $4::vector)
                        """,
                        rows,
                    )

            await store.aupdate_ingest_task(
                task_id, status="success", progress=PROGRESS_STAGES["done"],
                chunks_added=len(chunked_docs),
            )
            logger.info(f"异步导入完成: {task_id} ({filename}, {len(chunked_docs)} 块)")

        except Exception as e:
            logger.error(f"异步导入失败: {task_id} ({filename}): {e}")
            traceback.print_exc()
            try:
                await store.aupdate_ingest_task(
                    task_id, status="failed", progress=None,
                    error_msg=str(e)[:500],
                )
            except Exception:
                pass
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
