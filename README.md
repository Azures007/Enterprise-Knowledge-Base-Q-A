# 🏢 企业知识库问答系统 (RAG)

> 基于 **RAG（检索增强生成）** 技术构建的企业级智能问答系统。使用阿里云百炼 API（通义千问）作为底层大模型，ChromaDB 作为向量数据库，实现文档的智能解析、语义检索与自动问答。

---

## 📋 目录

- [功能特性](#-功能特性)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
- [依赖安装](#-依赖安装)
- [环境配置](#-环境配置)
- [文档导入](#-文档导入)
- [启动服务](#-启动服务)
- [前端开发指南](#-前端开发指南)
- [API 接口文档](#-api-接口文档)
- [命令行工具](#-命令行工具)
- [项目结构](#-项目结构)
- [常见问题](#-常见问题)
- [许可证](#-许可证)

---

## ✨ 功能特性

| 功能模块 | 说明 |
|---------|------|
| 📄 **文档解析** | 支持 PDF、Word (.docx)、PowerPoint (.pptx)、TXT、Markdown、JSON 等多种格式 |
| ✂️ **智能分块** | 自适应文本分块策略，优先在段落/句子边界切割，保持上下文完整性 |
| 🔍 **语义检索** | 基于通义千问 text-embedding-v3 的高质量向量嵌入 + ChromaDB 近邻搜索 |
| 💬 **智能问答** | 基于检索结果的上下文增强生成，回答附来源引用，有效减少幻觉 |
| ⚡ **流式输出** | 支持 SSE (Server-Sent Events) 流式响应，实时展示生成过程 |
| 🌐 **前端界面** | 基于 React 18 构建的现代化 Web 界面，支持实时流式对话、拖拽上传、暗色主题 |
| 🔄 **知识库管理** | 集合管理、文档增删、统计查询等功能 |
| 📡 **RESTful API** | 完整的 API 接口，方便与前端或其他服务集成 |
| 🖥️ **CLI 工具** | 终端可直接交互查询或批量导入文档 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       用户界面层 (React)                      │
│    ┌──────────────────┐  ┌──────────────────────────────┐   │
│    │   🌐 Web 界面     │  │      命令行工具               │   │
│    │  ┌──────────────┐│  │  ┌─────────┐ ┌──────────┐   │   │
│    │  │  对话式问答   ││  │  │ ingest │ │ query.py │   │   │
│    │  │  流式渲染     ││  │  └─────────┘ └──────────┘   │   │
│    │  │  拖拽上传     ││  └──────────────────────────────┘   │
│    │  │  来源引用     ││                                     │
│    │  └──────────────┘│                                     │
│    └──────────────────┘                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / SSE
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     API 服务层 (FastAPI)                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    RAG 管线                               │ │
│  │  1. 接收问题 → 2. 向量检索 → 3. 上下文组装 → 4. LLM 生成  │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
            │                    │
            ▼                    ▼
┌──────────────────┐  ┌──────────────────────────┐
│  向量数据库层      │  │   模型服务层 (阿里云百炼) │
│  ┌──────────────┐ │  │  ┌────────────────────┐ │
│  │   ChromaDB   │ │  │  │  通义千问 Qwen     │ │
│  │   (本地持久化) │ │  │  └────────┬───────────┘ │
│  └──────────────┘ │  │  ┌────────────────────┐ │
│                   │  │  │ text-embedding-v3  │ │
│                   │  │  └────────────────────┘ │
└──────────────────┘  └──────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│              文档处理管线                       │
│  PDF解析 → 文本提取 → 智能分块 → 向量化 → 入库  │
└──────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.10 或更高版本
- **操作系统**: Windows 10/11, macOS, Linux
- **网络**: 需能访问阿里云百炼 API (dashscope.aliyuncs.com)
- **磁盘空间**: 至少 500MB（用于依赖安装和向量数据库存储）

---

## 📦 依赖安装

### 第一步：创建虚拟环境（强烈推荐）

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 第二步：安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ 首次安装预计耗时 3-10 分钟，视网络状况而定。
>
> 如果遇到安装速度慢的问题，可以使用国内镜像源：
> ```bash
> pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
> ```

### 验证安装

```bash
python -c "import fitz; import docx; import chromadb; print('✅ 所有依赖安装成功')"
```

---

## 🔧 环境配置

### 快速配置

复制环境变量模板并直接使用（已包含 API 密钥）：

```bash
# Windows (cmd)
copy .env.example .env

# Windows (PowerShell)
cp .env.example .env

# macOS / Linux
cp .env.example .env
```

`.env` 文件内容说明：

```ini
# === 阿里云百炼 API 配置 ===
# 已预置 API 密钥，如无需修改可直接使用
BAILIAN_API_KEY=sk-4c889e3be9bf4b988005f7de49041851

# API 基础地址（OpenAI 兼容模式）
BAILIAN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# === 模型配置 ===
# 文本生成模型（通义千问系列）
LLM_MODEL_NAME=qwen-plus

# 文本嵌入模型
EMBEDDING_MODEL_NAME=text-embedding-v3

# === 向量数据库 ===
VECTOR_DB_PATH=./data/chroma_db

# === 文本分块参数 ===
CHUNK_SIZE=500        # 每块最大字符数
CHUNK_OVERLAP=100     # 块间重叠字符数

# === 检索配置 ===
RETRIEVAL_TOP_K=5     # 每次检索返回的文档块数量
```

### 获取自己的 API 密钥（可选）

如果想使用自己的 API 密钥：

1. 访问 [阿里云百炼控制台](https://bailian.console.aliyun.com/)
2. 在「模型广场」或「API-KEY 管理」中创建 API Key
3. 将 `.env` 文件中的 `BAILIAN_API_KEY` 替换为新密钥

---

## 📥 文档导入

### 准备文档

将需要导入的文档放入 `data/documents/` 目录：

```
data/documents/
├── 公司考勤管理制度.pdf
├── 产品技术白皮书.docx
├── 员工手册.txt
├── 会议纪要.pptx
└── 项目说明.md
```

### 批量导入目录

```bash
# 导入整个目录（含子目录递归）
python ingest.py -d data/documents/ -r
```

### 导入单个文件

```bash
# 导入指定文件
python ingest.py -f "data/documents/公司考勤管理制度.pdf"
```

### 清空并重新导入

```bash
# 先清空知识库，再导入目录
python ingest.py -d data/documents/ -r --reset
```

### 查看知识库状态

```bash
python ingest.py --stats
```

### 导入日志示例

```
📥 导入目录: data/documents/

[INFO] 在目录中找到 3 个文档待导入
--------------------------------------------------
[INFO] 正在解析文档: 公司考勤管理制度.pdf
[INFO]   解析完成，共 5 个原始文档段
[INFO]   正在进行文本分块...
[INFO]   分块完成，共 23 个文档块
[INFO]   正在计算向量并写入知识库...
[OK]   文件 '公司考勤管理制度.pdf' 导入成功！(23 个文档块)
--------------------------------------------------
[INFO] 正在解析文档: 员工手册.txt
...
[OK]   批量导入完成！
[INFO]   成功导入: 3/3 个文件
[INFO]   总文档块数: 67
```

---

## 🎯 启动服务

### 方案一：生产模式（推荐）

直接启动后端，FastAPI 会自动提供前端页面。

```bash
# 先构建前端
cd frontend && npm install && npm run build && cd ..

# 启动后端（自动提供前端页面 + API 接口）
python app.py
```

浏览器访问 **[http://localhost:8000](http://localhost:8000)** 即可使用完整界面。

### 方案二：前后端分离开发模式

后端和前端分别启动，支持热重载。

#### 启动后端

```bash
python app.py
# 或
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### 启动前端（新终端）

```bash
cd frontend
npm install   # 首次运行需要
npm run dev   # 开发模式，热重载
```

前端开发服务器运行在 **http://localhost:5173**，会自动代理 API 请求到后端。

### 验证服务运行

```bash
# 健康检查
curl http://localhost:8000/api/health

# 预期返回：
# {"status":"ok","service":"Enterprise Knowledge Base Q&A","version":"1.0.0"}
```

### 访问界面与文档

| 页面 | 地址 |
|------|------|
| **前端界面** | [http://localhost:8000](http://localhost:8000)（生产模式） |
| **前端开发服务器** | [http://localhost:5173](http://localhost:5173)（开发模式） |
| **Swagger API 文档** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **ReDoc** | [http://localhost:8000/redoc](http://localhost:8000/redoc) |

---

## 🎨 前端开发指南

### 技术栈

| 技术 | 用途 |
|------|------|
| **React 18** | UI 框架 |
| **Vite 6** | 构建工具（开发服务器热重载 + 生产构建） |
| **CSS 变量** | 亮色/暗色主题自适应（跟随系统） |
| **Fetch + SSE** | 与后端通信（含流式响应） |

### 目录结构

```
frontend/
├── index.html            # HTML 入口
├── package.json          # 依赖配置
├── vite.config.js        # Vite 配置（含 API 代理到 :8000）
├── dist/                 # 生产构建产物
└── src/
    ├── main.jsx          # React 入口
    ├── App.jsx           # 主组件（状态管理 + 布局）
    ├── App.css           # 全局样式（亮/暗主题）
    ├── services/
    │   └── api.js        # API 封装（7个接口 + SSE 流式）
    └── components/
        ├── Sidebar.jsx       # 侧边栏面板（统计、上传、操作）
        ├── ChatArea.jsx      # 对话区域（消息列表 + 输入框）
        └── MessageBubble.jsx # 消息气泡（含来源引用折叠）
```

### 开发模式（热重载）

```bash
# 终端 1: 启动后端
python app.py

# 终端 2: 启动前端开发服务器
cd frontend
npm run dev
```

前端运行在 **http://localhost:5173**，Vite 自动将 `/api` 请求代理到后端 `http://localhost:8000`。

### 生产构建

```bash
cd frontend
npm run build
```

构建产物在 `frontend/dist/` 目录。启动后端后，访问 **http://localhost:8000** 即可使用。

### 主题

界面自动跟随系统亮色/暗色主题。如需强制切换，在系统设置中更改即可。

### 组件说明

- **Sidebar**: 展示知识库统计（文档块数/查询次数）、集合列表、拖拽上传区域、操作按钮（清空对话/刷新/清空知识库）
- **ChatArea**: 消息列表区域（含欢迎页面引导）+ 底部输入框（Enter 发送、Shift+Enter 换行）+ 流式 SSE 渲染
- **MessageBubble**: 区分用户和 AI 消息样式，AI 消息下方可折叠展开来源引用列表

### API 对接说明

前端通过 `src/services/api.js` 统一调用后端 API。流式问答使用 SSE (Server-Sent Events) 实现逐字渲染：

```javascript
// 流式问答示例
import { streamQuery } from './services/api'

streamQuery(
  { question: '公司考勤制度是什么？' },
  {
    onChunk: (text) => { /* 逐字追加显示 */ },
    onSources: (sources) => { /* 显示来源 */ },
    onDone: (fullText) => { /* 完成回调 */ },
    onError: (err) => { /* 错误处理 */ },
  }
)
```

---

## 📡 API 接口文档

### 1. 知识库问答

```http
POST /api/query
Content-Type: application/json

{
    "question": "公司考勤制度规定上下班时间是什么？",
    "k": 5,
    "concise": false
}
```

响应示例：

```json
{
    "code": 0,
    "message": "success",
    "data": {
        "question": "公司考勤制度规定上下班时间是什么？",
        "answer": "根据公司考勤管理制度，工作时间如下：\n\n1. **工作日**：周一至周五\n2. **上班时间**：上午 9:00\n3. **下班时间**：下午 18:00\n4. **午休时间**：12:00 - 13:30\n\n（来源：公司考勤管理制度.pdf）",
        "sources": [
            {
                "filename": "公司考勤管理制度.pdf",
                "source": "data/documents/公司考勤管理制度.pdf",
                "page": 1,
                "score": 0.8923
            }
        ],
        "stats": {
            "retrieved_chunks": 5,
            "unique_sources": 2
        }
    }
}
```

### 2. 流式问答 (SSE)

```http
POST /api/query/stream
Content-Type: application/json

{
    "question": "公司考勤制度规定上下班时间是什么？"
}
```

SSE 响应格式：

```
data: {"type": "sources", "data": [...]}
data: {"type": "chunk", "data": "根据"}
data: {"type": "chunk", "data": "公司考勤管理制度"}
data: {"type": "chunk", "data": "，工作时间如下：..."}
data: {"type": "done", "data": "完整回答内容..."}
```

### 3. 上传文档

```http
POST /api/ingest
Content-Type: multipart/form-data

file: @考勤制度.pdf
```

### 4. 查看知识库集合

```http
GET /api/collections
```

### 5. 知识库统计

```http
GET /api/stats
```

### 6. 删除集合

```http
DELETE /api/collections/knowledge_base
```

### 7. 健康检查

```http
GET /api/health
```

### 使用 curl 测试

```bash
# 问答测试
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "公司考勤制度是什么？"}'

# 上传文档
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@data/documents/考勤制度.pdf"

# 查看统计
curl http://localhost:8000/api/stats
```

### 使用 Python requests 测试

```python
import requests

BASE_URL = "http://localhost:8000"

# 问答
resp = requests.post(f"{BASE_URL}/api/query", json={
    "question": "公司的考勤制度是什么？"
})
print(resp.json()["data"]["answer"])

# 上传文档
resp = requests.post(
    f"{BASE_URL}/api/ingest",
    files={"file": open("考勤制度.pdf", "rb")}
)
print(resp.json())
```

---

## 💻 命令行工具

### 交互式问答

进入交互模式后，可连续提问，类似聊天界面：

```bash
python query.py -i
```

效果：

```
╔══════════════════════════════════════════╗
║    企业知识库问答系统 - 交互模式         ║
╚══════════════════════════════════════════╝
  输入 exit 退出，输入 clear 清屏
  检索数量: 5

🧑 You: 公司的考勤制度是什么？

🤖 回答:

  根据公司考勤管理制度，工作时间如下：
  1. 工作日：周一至周五
  2. 上班时间：上午 9:00
  3. 下班时间：下午 18:00
  ...

📚 参考来源:
  - 公司考勤管理制度.pdf (得分: 0.892)

  检索了 5 个文档块 | 2 个来源
```

### 单次查询

```bash
python query.py -q "公司的薪资结构是怎样的？"
python query.py -q "产品技术架构" -k 10
python query.py -q "简单介绍" --concise
```

---

## 📁 项目结构

```
Enterprise-Knowledge-Base-Q-A/
├── app.py                          # FastAPI 应用入口
├── ingest.py                       # 文档导入 CLI 工具
├── query.py                        # 问答 CLI 工具
├── requirements.txt                # Python 依赖清单
├── .env.example                    # 环境变量模板
├── .gitignore                      # Git 忽略规则
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # 全局配置（类型安全，支持 .env）
│
├── src/
│   ├── __init__.py
│   │
│   ├── document_loader/            # 📄 文档解析模块
│   │   ├── __init__.py
│   │   └── loader.py               # 统一加载器（PDF/Word/PPTX/TXT/MD/JSON）
│   │
│   ├── text_processor/             # ✂️ 文本处理模块
│   │   ├── __init__.py
│   │   └── chunker.py              # 智能分块（段落/句子边界感知）
│   │
│   ├── embeddings/                 # 🔢 文本嵌入模块
│   │   ├── __init__.py
│   │   └── bailian_embedding.py    # 阿里云百炼 text-embedding-v3 封装
│   │
│   ├── vector_store/               # 🗄️ 向量数据库模块
│   │   ├── __init__.py
│   │   └── manager.py              # ChromaDB 管理（增删查改）
│   │
│   ├── llm/                        # 🧠 大模型模块
│   │   ├── __init__.py
│   │   └── bailian_llm.py          # 通义千问 API 封装（流式+非流式）
│   │
│   ├── rag/                        # 🔗 RAG 管线模块
│   │   ├── __init__.py
│   │   └── pipeline.py             # 检索+生成串联管线
│   │
│   ├── api/                        # 🌐 API 接口模块
│   │   ├── __init__.py
│   │   └── routes.py               # FastAPI 路由（7个接口）
│   │
│   └── utils/                      # 🛠️ 工具模块
│       ├── __init__.py
│       └── logger.py               # 统一日志配置
│
├── data/
│   ├── documents/                  # 📂 存放待导入的原始文档
│   ├── processed/                  # 💾 处理后的文本缓存
│   └── chroma_db/                  # 🗃️ ChromaDB 持久化数据
│
├── frontend/                       # 🌐 React 前端界面
│   ├── index.html                  # 页面入口
│   ├── package.json                # Node.js 依赖
│   ├── vite.config.js              # Vite 构建配置（含 API 代理）
│   ├── dist/                       # 构建产物（生产模式使用）
│   └── src/
│       ├── main.jsx                # React 入口
│       ├── App.jsx                 # 主组件（状态管理 + 布局）
│       ├── App.css                 # 全局样式（含暗色主题）
│       ├── services/
│       │   └── api.js              # API 封装层（7个接口 + SSE）
│       └── components/
│           ├── Sidebar.jsx         # 侧边栏（统计/上传/操作）
│           ├── ChatArea.jsx        # 对话区域（流式渲染）
│           └── MessageBubble.jsx   # 消息气泡（含来源引用）
│
└── logs/
    └── app.log                     # 📋 运行日志
```

---

## 🧪 完整使用流程

从零开始，完整的操作流程：

```bash
# 1. 克隆项目
git clone <项目地址>
cd Enterprise-Knowledge-Base-Q-A

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量（使用默认密钥）
cp .env.example .env

# 5. 将企业文档放入 data/documents/ 目录

# 6. 导入文档到知识库
python ingest.py -d data/documents/ -r

# 7. 启动 API 服务
python app.py

# 8. 打开另一个终端，测试问答
python query.py -i
```

---

## ❓ 常见问题

### Q: 导入文档时提示 "API 调用失败"

**原因**：API 密钥无效或网络不通。

**解决方法**：
1. 检查 `.env` 文件中 `BAILIAN_API_KEY` 是否正确
2. 确认网络能访问 `dashscope.aliyuncs.com`
3. 如需代理，设置环境变量：`export HTTP_PROXY=http://proxy:port`

### Q: 问答时返回 "当前知识库中没有找到相关信息"

**原因**：知识库为空，或文档未成功导入。

**解决方法**：
```bash
# 检查知识库状态
python ingest.py --stats

# 如果 total_chunks 为 0，需要导入文档
python ingest.py -d data/documents/ -r
```

### Q: 向量数据库文件损坏

**解决方法**：
```bash
# 清空并重新导入
python ingest.py --reset -d data/documents/ -r
```

### Q: 如何切换模型？

修改 `.env` 文件中的 `LLM_MODEL_NAME`：

```ini
# 使用更强模型
LLM_MODEL_NAME=qwen-max

# 更快响应
LLM_MODEL_NAME=qwen-turbo
```

### Q: 如何调整检索精度？

```ini
# 在 .env 中调整
CHUNK_SIZE=300         # 更小的块 = 更精确的检索
CHUNK_OVERLAP=50       # 减少重叠
RETRIEVAL_TOP_K=10     # 检索更多文档 = 更全面的上下文
```

### Q: 端口被占用怎么办？

```bash
# 使用其他端口启动
uvicorn app:app --host 0.0.0.0 --port 8080
```

---

## 📜 许可证

本项目仅供学习和研究使用。

---

## 🙏 致谢

- [阿里云百炼 (DashScope)](https://bailian.console.aliyun.com/) — 提供通义千问大模型和文本嵌入服务
- [ChromaDB](https://www.trychroma.com/) — 高性能向量数据库
- [FastAPI](https://fastapi.tiangolo.com/) — 现代化 Python Web 框架
- [LangChain](https://www.langchain.com/) — RAG 设计思路参考

---

> **提示**：如果在使用过程中遇到任何问题，请查看 `logs/app.log` 日志文件获取详细错误信息。
