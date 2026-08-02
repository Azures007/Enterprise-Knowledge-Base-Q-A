#!/usr/bin/env python3
"""
=============================================================================
知识库问答 CLI 工具 - 在终端中直接与知识库进行交互

支持单次查询和交互式对话模式，并显示检索到的参考来源。

使用示例:
    # 单次查询
    python query.py -q "公司的考勤制度是什么？"

    # 交互式对话模式
    python query.py -i

    # 简洁模式（简短回答）
    python query.py -q "考勤制度" --concise

    # 指定检索文档数量
    python query.py -q "薪资结构" -k 10
=============================================================================
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from src.rag import RAGPipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ==============================================================================
# 颜色与格式化工具
# ==============================================================================

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_answer(text: str):
    """打印回答内容"""
    print(f"\n{Colors.GREEN}{Colors.BOLD}🤖 回答:{Colors.RESET}\n")
    # 按段落格式化输出
    paragraphs = text.split("\n")
    for p in paragraphs:
        p = p.strip()
        if p:
            print(f"  {p}")
        else:
            print()


def print_sources(sources: list[dict]):
    """打印参考来源"""
    if not sources:
        return
    print(f"\n{Colors.CYAN}{Colors.BOLD}📚 参考来源:{Colors.RESET}")
    for src in sources:
        filename = src.get("filename", "未知")
        score = src.get("score", 0)
        source_path = src.get("source", "")
        print(f"  - {Colors.YELLOW}{filename}{Colors.RESET} "
              f"({Colors.DIM}得分: {score:.3f}{Colors.RESET})")
        # 如果存在页码信息，显示页码
        if src.get("page"):
            print(f"    └─ 第 {src['page']} 页")
        if src.get("slide"):
            print(f"    └─ 第 {src['slide']} 页幻灯片")
    print()


# ==============================================================================
# 交互模式
# ==============================================================================


def run_interactive(rag: RAGPipeline, k: int | None = None, concise: bool = False):
    """
    交互式问答模式。
    """
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}╔══════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}║    企业知识库问答系统 - 交互模式       ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}╚══════════════════════════════════════════╝{Colors.RESET}")
    print(f"  输入 {Colors.RED}exit{Colors.RESET} 退出，输入 {Colors.YELLOW}clear{Colors.RESET} 清屏")
    print(f"  检索数量: {Colors.CYAN}{k or 5}{Colors.RESET}")
    print()

    while True:
        try:
            question = input(f"{Colors.BOLD}🧑 You: {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.CYAN}再见！{Colors.RESET}")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print(f"{Colors.CYAN}再见！{Colors.RESET}")
            break

        if question.lower() in ("clear", "cls"):
            print("\033[2J\033[H", end="")  # 清屏
            continue

        # 显示"思考中"动画
        print(f"  {Colors.DIM}🤔 正在思考...{Colors.RESET}", end="\r")

        try:
            result = rag.query(
                question=question,
                k=k,
                stream=False,
                concise=concise,
            )

            # 清除"思考中"提示
            print(" " * 40, end="\r")

            answer = result["answer"]
            sources = result["sources"]

            print(f"{Colors.BOLD}🧑 You: {Colors.RESET}{question}")
            print_answer(answer)
            print_sources(sources)

            # 显示统计信息
            answer_type = result.get("answer_type", "general")
            type_label = {"general": "通用知识", "kb": "知识库", "hybrid": "混合知识"}
            stats = result.get("stats", {})

            # 构建状态行
            status_parts = []
            if answer_type == "kb" or answer_type == "hybrid":
                status_parts.append(f"📚 模式: {type_label.get(answer_type, answer_type)}")
                status_parts.append(f"检索了 {len(result['context'])} 个文档块")
                status_parts.append(f"{len(result['sources'])} 个来源")
            else:
                status_parts.append(f"💡 模式: {type_label.get(answer_type, answer_type)}")

            print(f"{Colors.DIM}  {' | '.join(status_parts)}{Colors.RESET}")
            print()

        except Exception as e:
            print(" " * 40, end="\r")
            print(f"\n{Colors.RED}❌ 错误: {e}{Colors.RESET}\n")


# ==============================================================================
# 主入口
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="企业知识库问答系统 - 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python query.py -q "公司考勤制度是什么？"
  python query.py -i
  python query.py -q "薪资结构" -k 10 --concise
        """,
    )

    # 查询模式
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-q", "--question",
        help="单次查询问题",
    )
    group.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="交互式对话模式",
    )

    # 选项
    parser.add_argument(
        "-k", type=int, default=5,
        help="检索文档块数量（默认: 5）",
    )
    parser.add_argument(
        "--concise",
        action="store_true",
        help="简洁回答模式",
    )

    args = parser.parse_args()

    # ---- 初始化 RAG 管线 ----
    try:
        print(f"{Colors.DIM}正在初始化 RAG 管线...{Colors.RESET}")
        rag = RAGPipeline()
        stats = rag.get_knowledge_base_stats()
        print(f"{Colors.DIM}知识库状态: {stats['total_chunks']} 个文档块{Colors.RESET}")
        print()
    except Exception as e:
        print(f"{Colors.RED}初始化失败: {e}{Colors.RESET}")
        sys.exit(1)

    # ---- 交互模式 ----
    if args.interactive:
        run_interactive(rag, k=args.k, concise=args.concise)
        return

    # ---- 单次查询 ----
    if args.question:
        question = args.question.strip()
        try:
            print(f"{Colors.DIM}🤔 正在生成回答...{Colors.RESET}")
            result = rag.query(
                question=question,
                k=args.k,
                stream=False,
                concise=args.concise,
            )

            answer = result["answer"]
            sources = result["sources"]
            answer_type = result.get("answer_type", "general")
            type_label = {"general": "💡 通用知识", "kb": "📚 知识库", "hybrid": "🔀 混合知识"}

            print(f"\n{Colors.BOLD}🧑 问题:{Colors.RESET} {question}")
            print(f"{Colors.DIM}回答模式: {type_label.get(answer_type, answer_type)}{Colors.RESET}")
            print_answer(answer)
            print_sources(sources)

        except Exception as e:
            print(f"{Colors.RED}❌ 查询失败: {e}{Colors.RESET}")
            sys.exit(1)
    else:
        parser.print_help()
        print(f"\n{Colors.YELLOW}提示: 使用 -q 提问 或 -i 进入交互模式{Colors.RESET}")


if __name__ == "__main__":
    main()
