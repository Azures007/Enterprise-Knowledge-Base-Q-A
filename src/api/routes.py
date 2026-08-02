"""
=============================================================================
API 路由模块

提供 RESTful API 接口，供前端或其他服务调用知识库问答功能。

接口列表:
    POST   /api/query            - 知识库问答
    POST   /api/query/stream     - 知识库问答（流式 SSE）
    POST   /api/ingest           - 上传并导入文档
    GET    /api/collections      - 查看知识库集合列表
    GET    /api/stats            - 查看知识库统计信息
    DELETE /api/collections/{name} - 删除指定集合
    GET    /api/health           - 健康检查
=============================================================================

使用方法（启动服务）:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import json
import os
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse

from src.document_loader import DocumentLoader
from src.embeddings import BailianEmbeddings
from src.llm import BailianLLM
from src.rag import RAGPipeline
from src.text_processor import TextChunker
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

# ==============================================================================
# 依赖注入：确保在整个 API 生命周期中复用同一实例
# ==============================================================================

_rag_pipeline: RAGPipeline | None = None
_document_loader: DocumentLoader | None = None
_text_chunker: TextChunker | None = None
_conversation_mgr: Any | None = None


def get_rag_pipeline() -> RAGPipeline:
    """获取或创建 RAG 管线单例"""
    global _rag_pipeline
    if _rag_pipeline is None:
        embedder = BailianEmbeddings()
        llm = BailianLLM()
        from src.vector_store import PGVectorStore
        vector_store = PGVectorStore(embedder)
        _rag_pipeline = RAGPipeline(
            embedder=embedder,
            llm=llm,
            vector_store=vector_store,
        )
    return _rag_pipeline


def get_document_loader() -> DocumentLoader:
    """获取文档加载器单例"""
    global _document_loader
    if _document_loader is None:
        _document_loader = DocumentLoader()
    return _document_loader


def get_text_chunker() -> TextChunker:
    """获取文本分块器单例"""
    global _text_chunker
    if _text_chunker is None:
        _text_chunker = TextChunker()
    return _text_chunker


def get_conversation_mgr() -> PGConversationManager:
    """获取对话管理器单例（复用 RAG 管线的 PGVectorStore）"""
    global _conversation_mgr
    if _conversation_mgr is None:
        from src.conversations import PGConversationManager
        # 复用 RAG 管线中的 PGVectorStore，避免重复创建导致集合重建
        rag = get_rag_pipeline()
        _conversation_mgr = PGConversationManager(vector_store=rag.vector_store)
    return _conversation_mgr


# ==============================================================================
# 路由定义
# ==============================================================================

router = APIRouter(prefix="/api", tags=["知识库问答"])


# ---------------------------------------------------------------
# 自动路由：根据问题判断最相关的集合
# ---------------------------------------------------------------

def _auto_route_collection(question: str, collections: list[str], rag) -> str | None:
    """
    根据用户问题自动选择最相关的知识库集合。

    如果无法确定，返回 None。
    """
    if len(collections) == 1:
        return collections[0]

    # 构造 prompt 让 LLM 判断
    collections_desc = "\n".join(f"- {c}" for c in collections)
    prompt = f"""根据用户的问题，从以下知识库集合中选择最相关的一个。
如果问题与任何一个集合都不相关，或者无法确定，请只返回「无法确定」四个字。
只返回集合名称或「无法确定」，不要其他文字。

可选的集合：
{collections_desc}

用户问题：{question}

最相关的集合名称："""

    try:
        answer = rag.llm.generate(
            prompt=prompt,
            system_prompt="你是一个知识库路由助手。分析用户问题，选择最匹配的集合名称，只返回名称本身。如果无法确定，返回「无法确定」。",
            max_tokens=50,
            temperature=0.1,
        )
        answer = answer.strip().strip('"').strip("'").strip()
        # 验证返回的集合名是否有效
        if answer in collections:
            logger.info(f"自动路由: '{question[:50]}...' → '{answer}'")
            return answer
    except Exception as e:
        logger.warning(f"自动路由异常: {e}")

    logger.info(f"自动路由无法确定: '{question[:50]}...'")
    return None


# ---------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------

@router.get("/health", summary="健康检查")
async def health_check():
    """
    服务健康检查接口，返回服务运行状态。
    """
    return {
        "status": "ok",
        "service": "Enterprise Knowledge Base Q&A",
        "version": "1.0.0",
    }


# ---------------------------------------------------------------
# 知识库问答
# ---------------------------------------------------------------

@router.post("/query", summary="知识库问答")
async def query_knowledge_base(
    request: Request,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    向知识库提问并获取回答。

    请求体格式 (JSON):
        {
            "question": "公司考勤制度是什么？",
            "k": 5,
            "concise": false,
            "collection": null
        }

    collection: 可选，指定查询的集合名称。不传则自动路由到最相关的集合。
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是有效的 JSON")

    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    k = body.get("k")
    concise = body.get("concise", False)
    filter_criteria = body.get("filter")
    collection_name = body.get("collection")

    # ---- 自动路由：确定查询哪个集合 ----
    all_collections = rag.vector_store.list_collections()
    if not all_collections:
        # 没有集合，直接返回空
        return {
            "code": 0,
            "message": "success",
            "data": {
                "question": question,
                "answer": "当前知识库为空，请先上传文档。",
                "sources": [],
                "answer_type": "general",
                "collection": None,
                "stats": {"retrieved_chunks": 0, "unique_sources": 0},
            },
        }

    if collection_name and collection_name in all_collections:
        # 用户指定了集合
        target_collection = collection_name
    elif len(all_collections) == 1:
        # 只有一个集合，直接使用
        target_collection = all_collections[0]
    else:
        # 多个集合，自动路由
        target_collection = _auto_route_collection(question, all_collections, rag)

    if target_collection is None:
        # 自动路由无法确定，提示用户手动选择
        coll_list = "、".join(all_collections)
        return {
            "code": 0,
            "message": "success",
            "data": {
                "question": question,
                "answer": f"无法确定您的问题属于哪个知识库。当前可用的集合有：{coll_list}。请手动选择对应的集合后重新提问。",
                "sources": [],
                "answer_type": "routing_failed",
                "collection": None,
                "stats": {"retrieved_chunks": 0, "unique_sources": 0},
            },
        }

    # 切换到目标集合
    rag.vector_store.switch_collection(target_collection)

    try:
        result = rag.query(
            question=question,
            k=k,
            stream=False,
            concise=concise,
            filter_criteria=filter_criteria,
        )
        return {
            "code": 0,
            "message": "success",
            "data": {
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "answer_type": result.get("answer_type", "general"),
                "collection": target_collection,
                "stats": {
                    "retrieved_chunks": len(result["context"]),
                    "unique_sources": len(result["sources"]),
                },
            },
        }
    except Exception as e:
        logger.error(f"问答接口异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答服务异常: {e}")


# ---------------------------------------------------------------
# 流式问答 (SSE)
# ---------------------------------------------------------------

@router.post("/query/stream", summary="流式知识库问答（SSE）")
async def query_knowledge_base_stream(
    request: Request,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    流式知识库问答，使用 Server-Sent Events (SSE) 协议。

    请求体格式 (JSON):
        {
            "question": "公司考勤制度是什么？",
            "k": 5,
            "concise": false
        }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是有效的 JSON")

    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    k = body.get("k")
    concise = body.get("concise", False)
    collection_name = body.get("collection")

    # ---- 自动路由：确定查询哪个集合 ----
    all_collections = rag.vector_store.list_collections()
    if not all_collections:
        async def empty_generator():
            yield f"data: {json.dumps({'type': 'meta', 'sources': [], 'answer_type': 'general'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'data': '当前知识库为空，请先上传文档。'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(empty_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    if collection_name and collection_name in all_collections:
        target_collection = collection_name
    elif len(all_collections) == 1:
        target_collection = all_collections[0]
    else:
        target_collection = _auto_route_collection(question, all_collections, rag)

    if target_collection is None:
        coll_list = "、".join(all_collections)
        async def routing_failed_generator():
            yield f"data: {json.dumps({'type': 'meta', 'sources': [], 'answer_type': 'routing_failed'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'data': f'无法确定您的问题属于哪个知识库。当前可用的集合有：{coll_list}。请手动选择对应的集合后重新提问。'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(routing_failed_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    # 切换到目标集合
    rag.vector_store.switch_collection(target_collection)

    async def event_generator():
        """SSE 事件生成器"""
        try:
            result = rag.stream_query(
                question=question,
                k=k,
                concise=concise,
            )
            sources = result["sources"]
            answer_type = result.get("answer_type", "general")
            answer_generator = result["answer"]

            # 发送来源信息和回答模式
            yield f"data: {json.dumps({'type': 'meta', 'sources': sources, 'answer_type': answer_type}, ensure_ascii=False)}\n\n"

            # 流式发送回答片段
            full_answer = ""
            for text_chunk in answer_generator:
                if text_chunk:
                    full_answer += text_chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'data': text_chunk}, ensure_ascii=False)}\n\n"

            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done', 'data': full_answer}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"流式问答异常: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------
# 文档导入
# ---------------------------------------------------------------

@router.get("/upload/token", summary="获取 OSS 直传签名 URL")
async def get_upload_token(
    filename: str = Query(..., description="文件名"),
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    获取 OSS 直传的预签名 URL，前端直接上传到 OSS 后可获得真实进度。

    使用流程：
        1. 前端调用此接口获取 upload_url
        2. 前端用 PUT 方法直接上传文件到 upload_url（可监听 XHR progress）
        3. 上传完成后，调用 POST /api/ingest/confirm 确认并解析
    """
    from src.storage import get_storage, OSSStorage
    storage = get_storage()

    if not isinstance(storage, OSSStorage):
        raise HTTPException(status_code=400, detail={
            "error_type": "oss_not_configured",
            "message": "当前存储后端不是 OSS，不支持直传",
            "suggestion": "请将 STORAGE_BACKEND 配置为 oss",
        })

    try:
        upload_info = storage.generate_upload_url(filename)
        return {
            "code": 0,
            "message": "success",
            "data": upload_info,
        }
    except Exception as e:
        logger.error(f"生成上传 token 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成上传 token 失败: {e}")


@router.post("/ingest/confirm", summary="确认 OSS 直传并导入文档")
async def confirm_upload(
    request: Request,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    loader: DocumentLoader = Depends(get_document_loader),
    chunker: TextChunker = Depends(get_text_chunker),
):
    """
    确认 OSS 直传完成，从 OSS 下载文件并导入知识库。

    请求体格式 (JSON):
        {
            "object_key": "knowledge_base_files/abc123.pdf",
            "filename": "考勤制度.pdf",
            "collection": "人事制度"
        }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是有效的 JSON")

    object_key = body.get("object_key", "").strip()
    filename = body.get("filename", "").strip() or object_key.split("/")[-1]
    collection_name = body.get("collection")

    if not object_key:
        raise HTTPException(status_code=400, detail="object_key 不能为空")

    # 切换到目标集合
    if collection_name:
        rag.vector_store.switch_collection(collection_name)

    # 从 OSS 下载文件
    from src.storage import get_storage, OSSStorage
    import tempfile

    storage = get_storage()
    if not isinstance(storage, OSSStorage):
        raise HTTPException(status_code=400, detail="当前存储后端不是 OSS")

    try:
        content = storage.get_object(object_key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"从 OSS 下载文件失败: {e}")

    file_size = len(content)
    file_ext = Path(filename).suffix.lower()
    logger.info(f"OSS 文件已下载: {object_key} ({file_size} bytes)")

    # 保存到临时文件并解析
    tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
    tmp_path = tmp_file.name
    tmp_file.write(content)
    tmp_file.close()

    try:
        # 计算哈希
        import hashlib
        content_hash = hashlib.sha256(content).hexdigest()

        # 查重（内容相同）
        existing = rag.vector_store.find_document_by_hash(content_hash)
        if existing is not None:
            os.unlink(tmp_path)
            raise HTTPException(status_code=409, detail={
                "error_type": "duplicate_document",
                "message": f"文件 '{filename}' 与已存在的文档内容重复",
                "suggestion": f"该文档已在 {existing['created_at'][:10]} 导入过",
            })

        # 查重（文件名相同）
        existing_name = rag.vector_store.find_document_by_filename(filename)
        if existing_name is not None:
            os.unlink(tmp_path)
            raise HTTPException(status_code=409, detail={
                "error_type": "duplicate_filename",
                "message": f"文件名 '{filename}' 已存在",
                "suggestion": "请使用不同的文件名上传",
            })

        # 解析文档
        raw_docs = loader.load_file(tmp_path)
        if not raw_docs:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail={
                "error_type": "empty_content",
                "message": f"文件 '{filename}' 解析后未提取到任何文本",
                "suggestion": "可能是扫描件，请安装 OCR 引擎",
            })

        # 修正 metadata 文件名
        for doc in raw_docs:
            if "metadata" in doc:
                doc["metadata"]["filename"] = filename
                doc["metadata"]["source"] = object_key

        # 记录文件元数据
        try:
            doc_id = rag.vector_store.add_document_record(
                filename=filename,
                file_type=file_ext.lstrip("."),
                file_size=file_size,
                content_hash=content_hash,
                storage_path=object_key,
                storage_backend="oss",
            )
        except Exception as e:
            logger.warning(f"记录文件元数据失败: {e}")
            doc_id = None

        # 分块和导入
        chunked_docs = chunker.split_documents(raw_docs)
        count = rag.vector_store.add_documents(chunked_docs, document_id=doc_id)

        return {
            "code": 0,
            "message": "success",
            "data": {
                "filename": filename,
                "chunks_added": count,
                "file_size": file_size,
                "storage": "oss",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档导入异常: {e}", exc_info=True)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"文档导入失败: {e}")


@router.post("/ingest", summary="上传并导入文档到知识库（传统方式）")
async def ingest_document(
    file: UploadFile = File(...),
    filename: str = Form(None),
    collection: str = Form(None),
    rag: RAGPipeline = Depends(get_rag_pipeline),
    loader: DocumentLoader = Depends(get_document_loader),
    chunker: TextChunker = Depends(get_text_chunker),
):
    """
    上传文档文件并将其导入知识库。

    支持的格式: pdf, docx, doc, docm, pptx, xlsx, xls,
                 jpg, jpeg, png, bmp, tiff, txt, md, json 等
    """
    # ================================================================
    # 前置校验
    # ================================================================

    # 1. 检查是否有文件
    if not file.filename:
        raise HTTPException(status_code=400, detail={
            "error_type": "no_file",
            "message": "没有选择文件",
            "suggestion": "请选择一个文件后再上传",
        })

    # 2. 检查文件扩展名
    allowed_extensions = set(loader.SUPPORTED_EXTENSIONS.keys())
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if not file_ext:
        raise HTTPException(status_code=400, detail={
            "error_type": "invalid_extension",
            "message": f"文件 '{file.filename}' 没有扩展名，无法识别文件类型",
            "suggestion": "请确保文件有正确的扩展名（如 .pdf, .docx, .txt）",
        })
    if file_ext not in allowed_extensions:
        # 按类型分组展示支持的格式
        type_groups = {
            "文档": [".pdf", ".docx", ".doc", ".docm", ".pptx"],
            "表格": [".xlsx", ".xls"],
            "图片（需 OCR）": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
            "文本/代码": [".txt", ".md", ".py", ".yaml", ".yml", ".html", ".htm", ".xml", ".csv", ".json"],
            "WPS": [".wps", ".et"],
        }
        support_list = "\n".join(f"  {k}: {', '.join(v)}" for k, v in type_groups.items())
        raise HTTPException(status_code=400, detail={
            "error_type": "unsupported_format",
            "message": f"不支持 '{file_ext}' 格式",
            "suggestion": f"支持的格式:\n{support_list}\n\n请将文件转换为支持的格式后重试",
        })

    # 3. 检查文件大小
    HUGE_FILE_LIMIT = 100 * 1024 * 1024
    LARGE_FILE_LIMIT = 50 * 1024 * 1024
    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(status_code=400, detail={
            "error_type": "empty_file",
            "message": f"文件 '{file.filename}' 为空",
            "suggestion": "请检查文件是否损坏，或重新导出后上传",
        })

    if file_size > HUGE_FILE_LIMIT:
        raise HTTPException(status_code=400, detail={
            "error_type": "file_too_large",
            "message": f"文件过大 ({file_size / 1024 / 1024:.1f}MB)",
            "suggestion": f"最大支持 {HUGE_FILE_LIMIT // 1024 // 1024}MB 的文件，当前文件 {file_size / 1024 / 1024:.1f}MB。请压缩或拆分后上传",
        })

    # ================================================================
    # 保存并解析
    # ================================================================

    import tempfile

    # 保存到临时文件用于解析
    tmp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
    tmp_path = tmp_file.name
    tmp_file.write(content)
    tmp_file.close()

    try:
        logger.info(f"文件已接收: {file.filename} ({file_size} bytes)")

        # ---- 计算内容哈希，检查是否重复 ----
        import hashlib
        content_hash = hashlib.sha256(content).hexdigest()
        logger.info(f"内容哈希: {content_hash[:16]}...")

        try:
            existing = rag.vector_store.find_document_by_hash(content_hash)
            logger.info(f"查重结果: {existing}")
        except AttributeError as e:
            logger.warning(f"查重方法不可用: {e}")
            existing = None

        if existing is not None:
            os.unlink(tmp_path)
            raise HTTPException(status_code=409, detail={
                "error_type": "duplicate_document",
                "message": f"文件 '{file.filename}' 与已存在的文档内容重复",
                "suggestion": (
                    f"该文档已在 {existing['created_at'][:10]} 导入过（"
                    f"原文件名: {existing['filename']}，{existing['file_size']} bytes）。\n"
                    f"如需重新导入，请先删除集合后重试。"
                ),
            })

        # ---- 检查文件名是否重复 ----
        # 确定最终使用的文件名（优先使用前端传的自定义文件名）
        final_filename = (filename or file.filename).strip()
        if not final_filename:
            final_filename = file.filename

        # 切换到目标集合
        if collection:
            rag.vector_store.switch_collection(collection)

        try:
            existing_name = rag.vector_store.find_document_by_filename(final_filename)
        except AttributeError:
            existing_name = None

        if existing_name is not None:
            os.unlink(tmp_path)
            raise HTTPException(status_code=409, detail={
                "error_type": "duplicate_filename",
                "message": f"文件名 '{final_filename}' 已存在",
                "suggestion": f"该文件名已在 {existing_name['created_at'][:10]} 导入过。请使用不同的文件名上传。",
            })

        # 解析文档
        raw_docs = loader.load_file(tmp_path)

        # 修正 metadata 中的文件名为原始文件名（解析时用的是临时文件名）
        for doc in raw_docs:
            if "metadata" in doc:
                doc["metadata"]["filename"] = final_filename
                doc["metadata"]["source"] = final_filename

        # 检查解析结果是否为空
        if not raw_docs:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail={
                "error_type": "empty_content",
                "message": f"文件 '{file.filename}' 解析后未提取到任何文本",
                "suggestion": (
                    "可能原因：\n"
                    "  - 扫描版 PDF/图片（需安装 OCR 引擎：pip install paddleocr）\n"
                    "  - 文档内容为纯图片/SmartArt\n"
                    "  - 文件已损坏"
                ),
            })

        # 上传到存储后端（OSS 或本地）
        from src.storage import get_storage
        storage = get_storage()
        storage_path = storage.save(file.filename, content)

        # 记录文件元数据到 PostgreSQL
        try:
            doc_id = rag.vector_store.add_document_record(
                filename=final_filename,
                file_type=file_ext.lstrip("."),
                file_size=file_size,
                content_hash=content_hash,
                storage_path=storage_path,
                storage_backend=settings.STORAGE_BACKEND,
            )
        except Exception as e:
            logger.warning(f"记录文件元数据失败（不影响解析）: {e}")
            doc_id = None

        # 检查是否有 warnings（如嵌入对象、SmartArt 等）
        warnings = []
        for doc in raw_docs:
            meta_warnings = doc.get("metadata", {}).get("warnings")
            if meta_warnings:
                warnings.extend(meta_warnings)

        # 文本分块
        chunked_docs = chunker.split_documents(raw_docs)

        # 导入知识库（关联 OSS 文件 ID）
        count = rag.vector_store.add_documents(
            chunked_docs,
            document_id=doc_id,
        )

        response_data = {
            "filename": file.filename,
            "chunks_added": count,
            "file_size": file_size,
            "storage": settings.STORAGE_BACKEND,
        }
        if warnings:
            response_data["warnings"] = list(set(warnings))

        return {
            "code": 0,
            "message": "success",
            "data": response_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档导入异常: {e}", exc_info=True)
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        error_msg = str(e)
        # 根据错误信息判断类型
        if "OCR" in error_msg and "安装" in error_msg:
            detail = {
                "error_type": "ocr_not_installed",
                "message": "需要 OCR 引擎来识别图片/扫描件中的文字",
                "suggestion": "请安装 OCR 引擎：pip install paddlepaddle paddleocr",
            }
        elif "LibreOffice" in error_msg:
            detail = {
                "error_type": "libreoffice_not_found",
                "message": f"文件 '{file.filename}' 需要 LibreOffice 转换",
                "suggestion": "请安装 LibreOffice (https://www.libreoffice.org/)，或将文件另存为支持的格式后重试",
            }
        elif "密码" in error_msg or "加密" in error_msg:
            detail = {
                "error_type": "encrypted",
                "message": f"文件 '{file.filename}' 已加密",
                "suggestion": "请先解密文件，或移除此密码保护后重新上传",
            }
        else:
            detail = {
                "error_type": "parse_error",
                "message": f"文件解析失败: {error_msg[:200]}",
                "suggestion": "请检查文件是否损坏、格式是否正确，或转换为其他格式后重试",
            }

        raise HTTPException(status_code=422, detail=detail)


# ---------------------------------------------------------------
# 文档管理（删除）
# ---------------------------------------------------------------

@router.get("/collections/{collection_name}/documents", summary="获取集合中的文档列表")
async def list_documents(
    collection_name: str,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """获取指定集合中的所有文档列表（含每个文档的分块数量）"""
    try:
        rag.vector_store.switch_collection(collection_name)
        docs = rag.vector_store.list_documents()
        return {"code": 0, "message": "success", "data": docs}
    except Exception as e:
        logger.error(f"获取文档列表异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {e}")


@router.delete("/documents/{doc_id}", summary="删除文档（含 OSS 原文件）")
async def delete_document(
    doc_id: int,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    删除指定文档及其所有分块，同时删除 OSS/本地存储中的原始文件。

    删除内容：
        - PostgreSQL chunks 表中的关联分块
        - PostgreSQL documents 表中的记录
        - OSS/本地存储中的原始文件
    """
    try:
        result = rag.vector_store.delete_document(doc_id, delete_storage=True)
        if result is None:
            raise HTTPException(status_code=404, detail="文档不存在")

        # 清空问答缓存
        from src.cache import qa_cache
        qa_cache.clear()

        return {
            "code": 0,
            "message": "success",
            "data": {
                "filename": result["filename"],
                "deleted_chunks": result["deleted_chunks"],
                "file_size": result["file_size"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除文档失败: {e}")


# ---------------------------------------------------------------
# 知识库管理
# ---------------------------------------------------------------

@router.get("/stats", summary="查看知识库统计信息")
async def get_stats(
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """获取知识库详细统计信息"""
    stats = rag.get_knowledge_base_stats()
    return {
        "code": 0,
        "message": "success",
        "data": stats,
    }


@router.get("/collections", summary="查看知识库集合列表")
async def list_collections(
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """列出所有知识库集合及其文档数量"""
    names = rag.vector_store.list_collections()
    # 获取每个集合的详细信息
    detailed = []
    for name in names:
        rag.vector_store.switch_collection(name)
        detailed.append({
            "name": name,
            "chunk_count": rag.vector_store.count(),
            "document_count": len(rag.vector_store.list_documents()),
        })
    return {
        "code": 0,
        "message": "success",
        "data": {
            "collections": detailed,
            "total": len(detailed),
        },
    }


@router.post("/collections", summary="创建新集合")
async def create_collection(
    request: Request,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """创建新的知识库集合"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是有效的 JSON")

    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="集合名称不能为空")

    if name in rag.vector_store.list_collections():
        raise HTTPException(status_code=409, detail=f"集合 '{name}' 已存在")

    # 创建集合（切换过去即自动创建）
    rag.vector_store.switch_collection(name)
    # 切换回默认集合
    rag.vector_store.switch_collection(settings.DEFAULT_COLLECTION)

    return {
        "code": 0,
        "message": "success",
        "data": {"name": name},
    }


@router.put("/collections/{collection_name}", summary="重命名集合")
async def rename_collection(
    collection_name: str,
    request: Request,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """重命名指定集合"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是有效的 JSON")

    new_name = body.get("name", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="新名称不能为空")

    if new_name in rag.vector_store.list_collections():
        raise HTTPException(status_code=409, detail=f"集合 '{new_name}' 已存在")

    ok = rag.vector_store.rename_collection(collection_name, new_name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"集合 '{collection_name}' 不存在")

    return {"code": 0, "message": "success", "data": {"old_name": collection_name, "new_name": new_name}}


@router.delete("/collections/{collection_name}", summary="删除指定集合（含 OSS 文件）")
async def delete_collection(
    collection_name: str,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """删除指定的知识库集合（含所有文档块和 OSS 原文件）"""
    try:
        rag.vector_store.switch_collection(collection_name)
        rag.vector_store.delete_collection()
        # 清空问答缓存
        from src.cache import qa_cache
        qa_cache.clear()
        # 切回一个存在的集合，防止 _ensure_collection 重建已删除的集合
        remaining = rag.vector_store.list_collections()
        if remaining:
            rag.vector_store.switch_collection(remaining[0])
        return {
            "code": 0,
            "message": f"集合 '{collection_name}' 已删除",
            "data": {},
        }
    except Exception as e:
        logger.error(f"删除集合失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除集合失败: {e}")


@router.get("/collections/{collection_name}/chunks", summary="查看集合中的文档块列表")
async def get_collection_chunks(
    collection_name: str,
    limit: int = 500,
    offset: int = 0,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    获取指定集合中所有文档块的列表，包含文本内容和元数据。

    参数:
        - collection_name: 集合名称
        - limit:  最大返回条数（默认 500）
        - offset: 分页偏移

    返回:
        每个文档块包含 id, content, metadata
    """
    try:
        # 切换到目标集合再读取
        rag.vector_store.switch_collection(collection_name)
        chunks = rag.get_collection_chunks(limit=limit, offset=offset)

        # 按源文件名分组
        grouped: dict[str, list] = {}
        for chunk in chunks:
            filename = chunk.get("metadata", {}).get("filename", "未知来源")
            if filename not in grouped:
                grouped[filename] = []
            grouped[filename].append(chunk)

        return {
            "code": 0,
            "message": "success",
            "data": {
                "total": len(chunks),
                "collection": collection_name,
                "chunks": chunks,
                "grouped": grouped,
            },
        }
    except Exception as e:
        logger.error(f"获取文档块列表异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档块列表失败: {e}")


# ---------------------------------------------------------------
# 对话管理
# ---------------------------------------------------------------

@router.get("/conversations", summary="获取所有对话列表")
async def list_conversations(
    mgr: ConversationManager = Depends(get_conversation_mgr),
):
    """获取所有历史对话列表（按更新时间倒序）"""
    try:
        convs = mgr.list_conversations()
        return {"code": 0, "message": "success", "data": convs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话列表失败: {e}")


@router.post("/conversations", summary="创建新对话")
async def create_conversation(
    mgr: ConversationManager = Depends(get_conversation_mgr),
):
    """创建一个新的空对话"""
    try:
        conv = mgr.create_conversation()
        return {"code": 0, "message": "success", "data": conv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建对话失败: {e}")


@router.delete("/conversations", summary="批量删除对话")
async def batch_delete_conversations(
    request: Request,
    mgr: ConversationManager = Depends(get_conversation_mgr),
):
    """批量删除指定对话（含所有消息）"""
    try:
        body = await request.json()
        ids = body.get("ids", [])
        if not ids or not isinstance(ids, list):
            raise HTTPException(status_code=400, detail="请提供要删除的对话 ID 列表")

        deleted = 0
        for conv_id in ids:
            if mgr.delete_conversation(conv_id):
                deleted += 1

        return {"code": 0, "message": f"成功删除 {deleted} 个对话", "data": {"deleted": deleted}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量删除失败: {e}")


@router.delete("/conversations/{conv_id}", summary="删除对话")
async def delete_conversation(
    conv_id: int,
    mgr: ConversationManager = Depends(get_conversation_mgr),
):
    """删除指定对话及其所有消息"""
    try:
        ok = mgr.delete_conversation(conv_id)
        if not ok:
            raise HTTPException(status_code=404, detail="对话不存在")
        return {"code": 0, "message": "对话已删除", "data": {}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除对话失败: {e}")


@router.put("/conversations/{conv_id}/title", summary="修改对话标题")
async def update_conversation_title(
    conv_id: int,
    request: Request,
    mgr: ConversationManager = Depends(get_conversation_mgr),
):
    """修改对话标题"""
    try:
        body = await request.json()
        title = body.get("title", "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="标题不能为空")
        ok = mgr.update_title(conv_id, title)
        if not ok:
            raise HTTPException(status_code=404, detail="对话不存在")
        return {"code": 0, "message": "success", "data": {}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"修改标题失败: {e}")


@router.get("/conversations/{conv_id}/messages", summary="获取对话消息列表")
async def get_messages(
    conv_id: int,
    mgr: ConversationManager = Depends(get_conversation_mgr),
):
    """获取指定对话的所有消息"""
    try:
        messages = mgr.get_messages(conv_id)
        return {"code": 0, "message": "success", "data": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取消息失败: {e}")


@router.post("/conversations/{conv_id}/messages", summary="添加消息到对话")
async def add_message(
    conv_id: int,
    request: Request,
    mgr: ConversationManager = Depends(get_conversation_mgr),
):
    """向对话中添加一条消息"""
    try:
        body = await request.json()
        role = body.get("role", "user")
        content = body.get("content", "")
        sources = body.get("sources")
        answer_type = body.get("answer_type")

        if not content:
            raise HTTPException(status_code=400, detail="消息内容不能为空")

        msg_id = mgr.add_message(conv_id, role, content, sources, answer_type)
        return {"code": 0, "message": "success", "data": {"id": msg_id}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加消息失败: {e}")


@router.put("/conversations/{conv_id}/messages/{msg_id}", summary="更新消息内容")
async def update_message(
    conv_id: int,
    msg_id: int,
    request: Request,
    mgr: ConversationManager = Depends(get_conversation_mgr),
):
    """更新消息内容（用于流式完成后补充完整内容和来源）"""
    try:
        body = await request.json()
        content = body.get("content", "")
        sources = body.get("sources")
        ok = mgr.update_message_content(msg_id, content, sources)
        if not ok:
            raise HTTPException(status_code=404, detail="消息不存在")
        return {"code": 0, "message": "success", "data": {}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新消息失败: {e}")
