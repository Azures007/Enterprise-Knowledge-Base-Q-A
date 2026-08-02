#!/usr/bin/env python3
"""
=============================================================================
文档导入工具 - 将本地文档批量导入知识库

支持递归导入目录中的所有文档，自动解析、分块、向量化后存入 ChromaDB。

使用示例:
    # 导入单个文件
    python ingest.py -f data/documents/公司考勤制度.pdf

    # 导入目录中的所有文档
    python ingest.py -d data/documents/

    # 递归导入目录（含子目录）
    python ingest.py -d data/documents/ -r

    # 查看文档块数量
    python ingest.py --stats

    # 清空知识库并重新导入
    python ingest.py -d data/documents/ --reset
=============================================================================
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.document_loader import DocumentLoader
from src.embeddings import BailianEmbeddings
from src.rag import RAGPipeline
from src.text_processor import TextChunker
from src.utils.logger import setup_logger
from src.vector_store import VectorStoreManager

logger = setup_logger(__name__)


# ==============================================================================
# 颜色输出工具
# ==============================================================================

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_info(msg: str):
    print(f"{Colors.CYAN}[INFO]{Colors.RESET} {msg}")


def print_success(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {msg}")


# ==============================================================================
# 导入流程
# ==============================================================================


def ingest_file(
    file_path: str,
    loader: DocumentLoader,
    chunker: TextChunker,
    rag: RAGPipeline,
) -> int:
    """
    导入单个文件到知识库。

    Returns:
        导入的文档块数量
    """
    path = Path(file_path)
    if not path.exists():
        print_error(f"文件不存在: {file_path}")
        return 0

    print_info(f"正在解析文档: {path.name}")
    raw_docs = loader.load_file(str(path))
    print_info(f"  解析完成，共 {len(raw_docs)} 个原始文档段")

    print_info(f"正在进行文本分块...")
    chunked_docs = chunker.split_documents(raw_docs)
    print_info(f"  分块完成，共 {len(chunked_docs)} 个文档块")

    print_info(f"正在计算向量并写入知识库...")
    count = rag.add_documents(chunked_docs)
    print_success(f"文件 '{path.name}' 导入成功！({count} 个文档块)")

    return count


def ingest_directory(
    directory: str,
    loader: DocumentLoader,
    chunker: TextChunker,
    rag: RAGPipeline,
    recursive: bool = True,
) -> int:
    """
    批量导入目录下的所有文档。

    Returns:
        总导入文档块数量
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print_error(f"目录不存在: {directory}")
        return 0

    pattern = "**/*" if recursive else "*"
    total_count = 0
    file_count = 0

    # 收集支持的文件
    supported_ext = loader.SUPPORTED_EXTENSIONS
    files = sorted([
        f for f in dir_path.glob(pattern)
        if f.is_file() and f.suffix.lower() in supported_ext
    ])

    if not files:
        print_warning(f"目录中没有找到支持的文档格式")
        print_info(f"支持的格式: {list(supported_ext.keys())}")
        return 0

    print_info(f"在目录中找到 {len(files)} 个文挡待导入")
    print("-" * 50)

    for file_path in files:
        try:
            count = ingest_file(str(file_path), loader, chunker, rag)
            if count > 0:
                total_count += count
                file_count += 1
        except Exception as e:
            print_error(f"导入失败 [{file_path.name}]: {e}")
            continue
        print("-" * 50)

    print()
    print_success(f"批量导入完成！")
    print_info(f"  成功导入: {file_count}/{len(files)} 个文件")
    print_info(f"  总文档块数: {total_count}")

    return total_count


# ==============================================================================
# 主入口
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="企业知识库 - 文档导入工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python ingest.py -f data/documents/report.pdf
  python ingest.py -d data/documents/
  python ingest.py -d data/documents/ --recursive
  python ingest.py --stats
  python ingest.py --reset
        """,
    )

    # 文件/目录参数
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-f", "--file",
        help="导入单个文件",
    )
    group.add_argument(
        "-d", "--directory",
        help="导入目录中的所有文档",
    )

    # 选项
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="递归导入子目录（与 -d 配合使用）",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清空知识库后重新导入",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="查看知识库统计信息",
    )

    args = parser.parse_args()

    # 初始化组件
    try:
        embedder = BailianEmbeddings()
        llm = None  # 导入时不需要 LLM
        vector_store = VectorStoreManager(embedder)
        rag = RAGPipeline(embedder=embedder, llm=None, vector_store=vector_store)
        # 兼容处理：RAGPipeline 需要 llm，但导入时用不到
        import src.llm.bailian_llm as bllm
        rag.llm = bllm.BailianLLM()

        loader = DocumentLoader()
        chunker = TextChunker()

    except Exception as e:
        print_error(f"初始化失败: {e}")
        sys.exit(1)

    # ---- 查看统计 ----
    if args.stats:
        stats = rag.get_knowledge_base_stats()
        print(f"\n{Colors.BOLD}📊 知识库统计{Colors.RESET}")
        print(f"  📦 集合: {stats['collections']}")
        print(f"  📄 文档块总数: {stats['total_chunks']}")
        print(f"  🔍 总查询次数: {stats['total_queries']}")
        return

    # ---- 重置知识库 ----
    if args.reset:
        print_warning("正在清空知识库...")
        vector_store.delete_collection()
        print_success("知识库已清空")

    # ---- 导入文件或目录 ----
    if args.file:
        print(f"\n{Colors.BOLD}📥 导入文件: {args.file}{Colors.RESET}\n")
        try:
            count = ingest_file(args.file, loader, chunker, rag)
            if count > 0:
                print_success(f"导入完成！共 {count} 个文档块")
        except Exception as e:
            print_error(f"导入失败: {e}")
            sys.exit(1)

    elif args.directory:
        print(f"\n{Colors.BOLD}📥 导入目录: {args.directory}{Colors.RESET}\n")
        try:
            total = ingest_directory(args.directory, loader, chunker, rag, args.recursive)
            if total > 0:
                print_success(f"导入完成！共 {total} 个文档块")
        except Exception as e:
            print_error(f"导入失败: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        print(f"\n{Colors.YELLOW}提示: 请使用 -f 或 -d 指定要导入的文件或目录{Colors.RESET}")


if __name__ == "__main__":
    main()
