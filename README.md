# 🏢 企业知识库问答系统 (RAG)

> 基于 **RAG（检索增强生成）** 技术构建的企业级智能问答系统。使用阿里云百炼 API（通义千问）作为底层大模型，支持 **PostgreSQL + pgvector** 与 **ChromaDB** 双向量存储后端，实现文档的智能解析、多集合并行语义检索、重排精排与自动问答。

---

## 📋 目录

- [功能特性](#-功能特性)
- [技术架构](#-技术架构)
- [快速开始](#-快速开始)
- [依赖安装](#-依赖安装)
- [环境配置](#-环境配置)
- [启动服务](#-启动服务)
- [文档导入](#-文档导入)
- [Docker 部署](#-docker-部署)
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
| 📄 **文档解析** | 支持 PDF、Word (.doc/.docx/.docm)、PowerPoint (.pptx)、Excel (.xls/.xlsx)、TXT、Markdown、HTML/XML/CSV/代码文件、JSON 及图片（OCR）等多种格式 |
| ✂️ **智能分块** | Markdown 标题感知语义分块（表格作为原子单元不拆散），普通文本优先在段落/句子边界切割，保持上下文完整 |
| 🔍 **混合检索** | 向量语义检索（text-embedding-v3）+ 关键词检索融合（向量 0.7 / 关键词 0.3），专有名词与编号类查询召回更准 |
| 🎯 **检索重排** | 两级重排：优先 `bge-reranker` 交叉编码器精排（可选装），不可用时自动降级为「向量分 + 中文关键词重合度」轻量重排，带关键词否决机制 |
| 🧲 **相关性过滤** | 单块分数阈值（0.35）+ 文档级阈值双重过滤，弱相关文档整篇剔除，有效减少幻觉 |
| 💬 **智能问答** | 混合回答策略（知识库 / 通用知识 / 二者结合），回答内嵌 [N] 引用标注，附来源列表 |
| 👍 **用户反馈** | 对每条 AI 回答点赞/点踩（点踩可选填原因），状态持久化，为质量优化积累数据 |
| 📊 **查询审计** | 记录每次问答的用户/问题/来源/Token 用量/延迟/缓存命中，管理员可查询与汇总统计 |
| 💭 **多轮对话** | 对话持久化、历史 token 预算截断、LLM 问题重写（把指代问题改写成独立完整问题）、相关问题推荐 |
| 🛠️ **工具调用** | Function Calling 支持：LLM 自主调用工具（列集合、检索知识库、查统计、时间/计算），Agentic 检索按需跨集合查询 |
| ⚡ **流式输出** | 支持 SSE (Server-Sent Events) 流式响应，实时展示生成过程 |
| 📚 **知识库管理** | 多集合（增删改查）、集合归属权限、文档增删、分块查看、统计查询 |
| 👥 **用户与认证** | JWT + 静态 API Key 双认证，用户管理（管理员创建/删除/重置密码），用户级集合隔离 |
| 🚦 **稳定性保障** | 分级限流（IP/全局/LLM 配额保护）、LRU 问答缓存（按来源文件精准失效）、异步大文件导入 |
| 📡 **RESTful API** | 完整的 API 接口，方便与前端或其他服务集成 |
| 🖥️ **CLI 工具** | 终端可直接交互查询或批量导入文档 |

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                       用户界面层 (React)                      │
│    ┌──────────────────┐  ┌──────────────────────────────┐   │
│    │   🌐 Web 界面     │  │      命令行工具               │   │
│    │  ┌──────────────┐│  │  ┌────────────┐ ┌────────┐  │   │
│    │  │  对话式问答   ││  │  │  ingest.py │ │query.py│  │   │
│    │  │  流式渲染     ││  │  └────────────┘ └────────┘  │   │
│    │  │  会话管理     ││  │                             │   │
│    │  │  集合管理     ││  └──────────────────────────────┘   │
│    │  │  用户管理     ││                                     │
│    │  └──────────────┘│                                     │
│    └──────────────────┴─────────────────────────────────────┘
│                            ▲  │ HTTP / SSE
│                   登录/JWT  │  │ 认证保护
└────────────────────────────┼──┼─────────────────────────────┘
                             │  ▼
┌──────────────────────────────────────────────────────────────┐
│                     API 服务层 (FastAPI)                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                RAG 管线（混合问答策略）                    │ │
│  │  问题重写 → 多集合并行检索 → 重排 → 相关性过滤 → LLM 生成  │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │   中间件：认证(JWT/API Key) · 分级限流 · CORS · 全局异常   │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────┬───────────────────────────────┬───────────────┘
               ▼                               ▼
┌──────────────────────┐      ┌──────────────────────────────┐
│  向量数据库层          │      │   模型服务层 (阿里云百炼)      │
│  ┌──────────────────┐ │      │  ┌────────────────────────┐ │
│  │  PostgreSQL      │ │      │  │  通义千问 Qwen         │ │
│  │  + pgvector      │ │      │  └──────────┬─────────────┘ │
│  │  （推荐，统一存储） │ │      │  ┌────────────────────────┐ │
│  └──────────────────┘ │      │  │ text-embedding-v3      │ │
│  ┌──────────────────┐ │      │  └────────────────────────┘ │
│  │  ChromaDB        │ │      │  ┌────────────────────────┐ │
│  │  （嵌入式备选）    │ │      │  │ bge-reranker（可选）    │ │
│  └──────────────────┘ │      │  └────────────────────────┘ │
└──────────┬───────────┘      └──────────────────────────────┘
           │
┌──────────────────────────────────────────────────────────────┐
│   基础设施：PostgreSQL(用户/对话/文档/分块) · Redis(限流，     │
│              可选降级内存) · OSS/S3(原始文件，可选) · 本地磁盘  │
└──────────────────────────────────────────────────────────────┘
```

### 核心数据流

```
文档导入:  上传 → 扩展名/大小校验 → 内容哈希查重 → 解析 → 智能分块
          → 向量化 → 入库（文档记录与分块同事务，杜绝僵尸记录）

知识问答:  问题 → (多轮历史) → 问题重写 → 多集合并行检索
          → 候选放宽(2×k) → 重排精排 → 相关性过滤 → 上下文组装
          → LLM 生成（回答内嵌 [N] 引用） → 来源列表 + 相关问题推荐
```

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.10 或更高版本
- **Node.js**: 18+（构建前端界面）
- **操作系统**: Windows 10/11, macOS, Linux
- **数据库**: PostgreSQL 12+（推荐，需启用 pgvector 扩展）；或用内置 ChromaDB（零配置）
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

### 可选增强依赖

| 功能 | 安装命令 | 说明 |
|------|---------|------|
| OCR 识别（扫描件/图片） | `pip install paddlepaddle paddleocr`（推荐，中文效果好）或 `pip install easyocr` | 不装则扫描版 PDF / 图片无法提取文字 |
| .doc 旧格式解析 | `pip install textract` 或安装 LibreOffice | 否则 .doc 需经 LibreOffice 转换 |
| 交叉编码器重排 | `pip install sentence-transformers` | 不装自动降级为轻量关键词重排 |
| 阿里云 OSS 存储 | `pip install oss2` | 不装则使用本地磁盘存储 |
| Redis 限流 | `pip install redis` | 不装自动降级为内存限流 |

### 准备数据库（使用 pgvector 时）

```sql
-- PostgreSQL 中创建数据库
CREATE DATABASE knowledge_base;

-- 启用 pgvector 扩展（系统首次启动时也会自动尝试创建）
CREATE EXTENSION IF NOT EXISTS vector;
```

> 数据库表结构（collections / documents / chunks / conversations / messages / users / ingest_audit_log / query_audit_log）会在服务首次启动时自动创建，无需手动建表。`messages` 表含反馈列（feedback / feedback_comment / feedback_at）。

### 验证安装

```bash
python -c "import fitz; import docx; import chromadb; import asyncpg; print('✅ 所有依赖安装成功')"
```

---

## 🔧 环境配置

### 快速配置

复制环境变量模板并填写：

```bash
# Windows (cmd)
copy .env.example .env

# Windows (PowerShell)
cp .env.example .env

# macOS / Linux
cp .env.example .env
```

`.env` 文件关键配置项：

```ini
# === 阿里云百炼 API 配置 ===
BAILIAN_API_KEY=sk-your-bailian-api-key-here   # 必填
BAILIAN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# === 模型配置 ===
LLM_MODEL_NAME=qwen-plus          # 可选 qwen-max / qwen-turbo
EMBEDDING_MODEL_NAME=text-embedding-v3

# === 向量数据库 ===
VECTOR_STORE_TYPE=pg              # pg（推荐）| chroma
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/knowledge_base
VECTOR_DB_PATH=./data/chroma_db   # 仅 chroma 后端生效

# === 文档处理参数 ===
CHUNK_SIZE=500        # 每块最大字符数
CHUNK_OVERLAP=100     # 块间重叠字符数

# === 检索配置 ===
RETRIEVAL_TOP_K=5     # 最终返回的文档块数量
RETRIEVAL_CANDIDATE_K=20  # 重排前的候选块数量（粗召回）
RERANK_ENABLED=true   # 启用重排（bge-reranker，不可用自动降级）

# === 对象存储（原始文件）===
STORAGE_BACKEND=local             # local | oss | s3
# OSS_ENDPOINT=...
# OSS_BUCKET=...
# OSS_ACCESS_KEY=...
# OSS_SECRET_KEY=...

# === 认证配置（生产环境建议启用）===
AUTH_ENABLED=true                 # 开启后所有 /api 接口需认证
JWT_SECRET_KEY=change-me-in-production  # 生产环境务必改为随机长字符串
AUTH_ADMIN_USERNAME=admin         # 管理员账号（首次启动自动创建）
AUTH_ADMIN_PASSWORD=admin123      # 管理员密码（生产环境务必修改）
API_KEYS=                         # 静态 API Key 列表（逗号分隔，可选）

# === 缓存配置 ===
CACHE_MAX_ENTRIES=10000           # LRU 缓存上限
CACHE_QA_TTL=3600                 # 问答缓存 TTL（秒）

# === 工具调用配置（Function Calling / Agentic 检索）===
TOOL_CALLING_ENABLED=false        # 启用后 LLM 自主调用工具检索（默认关闭）
TOOL_CALLING_MAX_ROUNDS=4         # 工具循环最大轮数（防死循环）
TOOL_CALLING_RESULT_LIMIT=2000    # 工具结果截断字符数
```

完整配置项见 [config/settings.py](config/settings.py)。

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

浏览器访问 **[http://localhost:8000](http://localhost:8000)** 即可使用完整界面。默认管理员账号 `admin / admin123`（首次登录后建议修改密码）。

### 方案二：前后端分离开发模式

后端和前端分别启动，支持热重载。

```bash
# 终端 1：启动后端
python app.py
# 或
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

```bash
# 终端 2：启动前端开发服务器（新终端）
cd frontend
npm install   # 首次运行需要
npm run dev   # 开发模式，热重载
```

前端开发服务器运行在 **http://localhost:5173**，会自动代理 API 请求到后端。

### 验证服务运行

```bash
curl http://localhost:8000/api/health
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

## 📥 文档导入

### 通过 Web 界面上传

登录后在侧边栏的**上传区域**拖拽或选择文件即可导入。系统会自动：

1. 校验格式与大小（支持格式见下表，最大 100MB）
2. 内容哈希查重（重复文件返回 409 提示已导入时间）
3. 解析 → 分块 → 向量化 → 入库
4. 文档归入当前用户的**个人集合**（管理员归入「知识库」）

> **大文件自动走异步导入**：超过 10MB 的文件提交到后台任务队列，前端展示实时进度条（下载→查重→解析→分块→嵌入→完成），不会阻塞其他请求。
>
> **OSS 直传**（配置 `STORAGE_BACKEND=oss` 后）：前端先获取预签名 URL 直接上传到 OSS（可监听真实上传进度），再确认导入。

### 支持的格式

| 类型 | 扩展名 |
|------|--------|
| 文档 | .pdf, .docx, .doc, .docm, .pptx |
| 表格 | .xlsx, .xls |
| 图片（需 OCR） | .jpg, .jpeg, .png, .bmp, .tiff, .tif |
| 文本/代码 | .txt, .md, .py, .yaml, .yml, .html, .htm, .xml, .csv, .json |

### 命令行批量导入

将文档放入 `data/documents/` 目录后：

```bash
# 导入整个目录（含子目录递归）
python ingest.py -d data/documents/ -r

# 导入单个文件
python ingest.py -f "data/documents/公司考勤管理制度.pdf"

# 清空并重新导入
python ingest.py -d data/documents/ -r --reset

# 查看知识库状态
python ingest.py --stats
```

---

## 🐳 Docker 部署

### 一键部署（后端 + 前端单容器）

```bash
# 1. 复制并配置 .env
cp .env.example .env

# 2. 构建并启动（多阶段构建：Node 构建前端 → Python 运行后端）
docker compose up --build
```

服务启动于 **http://localhost:8000**。

> 注意：docker-compose.yml 默认持久化 ChromaDB 数据目录。若使用 PostgreSQL + pgvector 后端（`VECTOR_STORE_TYPE=pg`），需额外配置数据库容器或使用外部 PostgreSQL，并确保 `.env` 中的 `DATABASE_URL` 指向可达地址。

### 手动构建

```bash
docker build -t enterprise-kb .
docker run -d -p 8000:8000 --env-file .env enterprise-kb
```

### 数据持久化（docker-compose）

| 卷 | 用途 |
|------|------|
| `./data/chroma_db` | 向量数据库（知识库内容） |
| `./data/documents` | 上传/导入的原始文档 |
| `./data/processed` | 处理后文本缓存 |
| `./logs` | 应用日志 |

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
    ├── App.jsx           # 主组件（认证 + 状态管理 + 布局）
    ├── App.css           # 全局样式（亮/暗主题）
    ├── services/
    │   └── api.js        # API 封装（30+ 接口 + SSE 流式）
    └── components/
        ├── LoginPage.jsx           # 登录页
        ├── Sidebar.jsx             # 侧边栏（统计、上传、集合、操作）
        ├── ChatArea.jsx            # 对话区域（消息列表 + 输入框）
        ├── MessageBubble.jsx       # 消息气泡（来源引用折叠、相关问题推荐、点赞/点踩）
        ├── ConversationPanel.jsx   # 会话管理面板（新建/删除/改标题）
        ├── CollectionModal.jsx     # 集合管理弹窗（创建/重命名/删除/查看分块）
        ├── UploadQueue.jsx         # 上传队列（异步进度条）
        ├── UserManagementModal.jsx # 用户管理弹窗（管理员）
        ├── ChangePasswordModal.jsx # 修改密码弹窗
        └── AuditModal.jsx          # 查询审计弹窗（管理员，汇总卡片+记录表格）
```

### 组件说明

- **LoginPage**：登录界面，支持用户名密码登录，JWT 令牌本地持久化，401 自动登出
- **Sidebar**：知识库统计（文档块数/查询次数）、集合列表与切换、拖拽上传区域、操作按钮
- **ChatArea**：消息列表 + 底部输入框（Enter 发送、Shift+Enter 换行）+ 流式 SSE 渲染
- **MessageBubble**：区分用户和 AI 消息样式，AI 消息下方可折叠展开来源引用列表，展示相关问题推荐，支持点赞/点踩反馈（点踩可填原因）
- **ConversationPanel**：多轮会话管理，新建对话、切换、删除、重命名
- **CollectionModal**：集合管理，创建/重命名/删除集合、查看集合内文档与分块
- **UploadQueue**：上传任务队列，展示同步/异步/OSS 直传进度
- **AuditModal**：查询审计管理弹窗，展示汇总卡片（总查询/缓存率/延迟/Token）、最近查询表格（点击行展开回答与来源）、热门问题 TopN

### 流式问答示例

前端通过 `src/services/api.js` 统一调用后端 API。流式问答使用 SSE (Server-Sent Events) 实现逐字渲染：

```javascript
import { streamQuery } from './services/api'

streamQuery(
  { question: '公司考勤制度是什么？', k: 5, concise: false, collection: null },
  {
    onChunk: (text) => { /* 逐字追加显示 */ },
    onMeta: ({ sources, answer_type }) => { /* 显示来源和回答模式 */ },
    onDone: (fullText) => { /* 完成回调 */ },
    onError: (err) => { /* 错误处理 */ },
  }
)
```

---

## 📡 API 接口文档

### 认证

所有接口（除登录与健康检查外）在 `AUTH_ENABLED=true` 时需携带令牌，二选一：

```http
Authorization: Bearer <jwt_token>      # 通过 /api/auth/login 获取
X-API-Key: <static_api_key>            # 在 .env 的 API_KEYS 中配置
```

### 1. 登录

```http
POST /api/auth/login
Content-Type: application/json

{"username": "admin", "password": "admin123"}
```

响应返回 JWT 令牌：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "token": "eyJhbGciOi...",
    "token_type": "bearer",
    "expires_in": 604800,
    "username": "admin",
    "is_admin": true
  }
}
```

### 2. 知识库问答

```http
POST /api/query
Content-Type: application/json
Authorization: Bearer <token>

{
    "question": "公司考勤制度规定上下班时间是什么？",
    "k": 5,
    "concise": false,
    "conversation_id": null
}
```

响应示例：

```json
{
    "code": 0,
    "message": "success",
    "data": {
        "question": "公司考勤制度规定上下班时间是什么？",
        "answer": "根据公司考勤管理制度，工作时间如下：\n\n1. **工作日**：周一至周五\n2. **上班时间**：上午 9:00\n3. **下班时间**：下午 18:00\n\n（来源：公司考勤管理制度.pdf[1]）",
        "sources": [
            {
                "index": 1,
                "filename": "公司考勤管理制度.pdf",
                "source": "data/documents/公司考勤管理制度.pdf",
                "page": 1,
                "score": 0.8923,
                "chunks": [{"content": "...", "score": 0.8923, "chunk_index": 0}]
            }
        ],
        "answer_type": "kb",
        "collection": "知识库",
        "stats": {"retrieved_chunks": 5, "unique_sources": 2},
        "related_questions": ["公司的迟到处理办法是什么？", "加班补贴如何计算？"]
    }
}
```

> `answer_type` 取值：`kb`（知识库回答）、`general`（通用知识）、`hybrid`（混合）。

### 3. 流式问答 (SSE)

```http
POST /api/query/stream
Content-Type: application/json

{"question": "公司考勤制度规定上下班时间是什么？", "conversation_id": null}
```

SSE 响应格式：

```
data: {"type": "meta", "sources": [...], "answer_type": "kb"}
data: {"type": "chunk", "data": "根据"}
data: {"type": "chunk", "data": "公司考勤管理制度"}
data: {"type": "chunk", "data": "，工作时间如下：..."}
data: {"type": "done", "data": "完整回答内容..."}
```

### 4. 上传文档

```http
POST /api/ingest
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: @考勤制度.pdf
filename: 考勤制度.pdf          # 可选，自定义文件名
collection: 知识库              # 可选，指定集合（仅限有权限的集合）
```

**异步导入**（>10MB 大文件）与 **OSS 直传**：

```http
POST /api/ingest/async          # 立即返回 task_id
GET  /api/ingest/tasks/{task_id}  # 轮询任务进度（pending→processing→success/failed）
GET  /api/upload/token?filename=xx.pdf  # 获取 OSS 直传预签名 URL
POST /api/ingest/confirm        # 确认 OSS 直传完成并导入
```

### 5. 用户与认证管理

```http
POST   /api/auth/change-password          # 修改自己的密码
GET    /api/users                         # 用户列表（管理员）
POST   /api/users                         # 创建用户（管理员）
DELETE /api/users/{user_id}               # 删除用户（管理员）
POST   /api/users/{user_id}/reset-password  # 重置用户密码（管理员）
```

### 6. 集合管理

```http
GET    /api/collections                            # 集合列表（含文档数/分块数）
POST   /api/collections                            # 创建集合（归属当前用户）
PUT    /api/collections/{name}                     # 重命名集合（归属者或管理员）
DELETE /api/collections/{name}                     # 删除集合（含 OSS 文件）
GET    /api/collections/{name}/documents           # 集合内文档列表
GET    /api/collections/{name}/chunks              # 集合内分块列表（按源文件分组）
DELETE /api/documents/{doc_id}                     # 删除单个文档（含 OSS 原文件）
```

### 7. 对话管理

```http
GET    /api/conversations                          # 对话列表（当前用户）
POST   /api/conversations                          # 创建对话
DELETE /api/conversations/{conv_id}                # 删除对话
DELETE /api/conversations                          # 批量删除（body: {"ids": [1,2]}）
PUT    /api/conversations/{conv_id}/title          # 修改对话标题
GET    /api/conversations/{conv_id}/messages       # 查看对话消息（含反馈状态）
POST   /api/conversations/{conv_id}/messages       # 添加消息
PUT    /api/conversations/{conv_id}/messages/{msg_id}  # 更新消息内容/来源
POST   /api/conversations/{conv_id}/messages/{msg_id}/feedback  # 消息反馈
```

**消息反馈**：对 AI 回答点赞/点踩，body: `{"feedback": 1}`（1=赞，-1=踩，0=清除），点踩可带 `"comment": "原因"`。反馈状态随消息查询返回（`feedback` / `feedback_comment` / `feedback_at` 字段）。

### 8. 查询审计（管理员）

```http
GET /api/audit/queries?limit=50&offset=0&username=admin   # 审计记录列表（按时间倒序）
GET /api/audit/summary                                     # 审计汇总统计
```

`/api/audit/summary` 返回：总查询数、缓存命中率、平均延迟、总 Token（输入/输出/合计）、回答类型分布（kb/hybrid/general）、热门问题 TopN。每次问答（含流式）都会记录用户、问题、回答、来源、Token 真实用量、延迟、是否缓存命中；失败请求也会记录 `status=failed`。由 `AUDIT_ENABLED` 开关控制。

### 9. 统计与健康检查

```http
GET /api/stats     # 知识库统计（集合数、文档块数、查询次数等）
GET /api/health    # 健康检查
```

### 使用 curl 测试

```bash
# 登录获取 token
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | python -c "import sys,json;print(json.load(sys.stdin)['data']['token'])")

# 问答测试
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "公司考勤制度是什么？"}'

# 上传文档
curl -X POST http://localhost:8000/api/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@data/documents/考勤制度.pdf"

# 查看统计
curl http://localhost:8000/api/stats -H "Authorization: Bearer $TOKEN"
```

---

## 💻 命令行工具

### 交互式问答

进入交互模式后，可连续提问，类似聊天界面：

```bash
python query.py -i
```

### 单次查询

```bash
python query.py -q "公司的薪资结构是怎样的？"
python query.py -q "产品技术架构" -k 10
python query.py -q "简单介绍" --concise
```

### 批量导入

```bash
python ingest.py -d data/documents/ -r   # 递归导入目录
python ingest.py -f data/documents/报告.pdf  # 导入单个文件
python ingest.py --stats                 # 查看知识库统计
python ingest.py --reset                 # 清空知识库
```

---

## 📁 项目结构

```
Enterprise-Knowledge-Base-Q-A/
├── app.py                          # FastAPI 应用入口（lifespan 生命周期、中间件、静态托管）
├── ingest.py                       # 文档导入 CLI 工具
├── query.py                        # 问答 CLI 工具
├── requirements.txt                # Python 依赖清单
├── .env.example                    # 环境变量模板
├── Dockerfile                      # 多阶段构建（Node 前端 + Python 后端）
├── docker-compose.yml              # 一键部署编排
│
├── config/
│   └── settings.py                 # 全局配置（pydantic-settings，类型安全）
│
├── src/
│   ├── api/
│   │   ├── routes.py               # FastAPI 路由（31 个接口）
│   │   └── models.py               # Pydantic 请求/响应模型
│   ├── document_loader/            # 📄 文档解析模块
│   │   ├── loader.py               # 统一加载器（PDF/Word/PPTX/Excel/TXT/图片等）
│   │   └── ocr.py                  # OCR 识别（扫描件/图片）
│   ├── text_processor/
│   │   └── chunker.py              # 智能分块（Markdown 标题感知 + 段落/句子边界）
│   ├── embeddings/
│   │   └── bailian_embedding.py    # 阿里云百炼 text-embedding-v3 封装
│   ├── vector_store/
│   │   ├── pg_manager.py           # PostgreSQL + pgvector 存储（推荐，统一存储）
│   │   └── manager.py              # ChromaDB 存储（备选）
│   ├── llm/
│   │   └── bailian_llm.py          # 通义千问 API 封装（流式+非流式）
│   ├── rag/
│   │   └── pipeline.py             # RAG 管线（问题重写/检索/重排/生成/推荐/Agentic 工具模式）
│   ├── tools/                      # 🛠️ 工具调用模块（Function Calling）
│   │   ├── registry.py             # 工具注册表 + 工具循环执行器
│   │   ├── tools.py                # 内置工具（时间/计算/列集合/检索知识库/统计/审计）
│   │   └── weather.py              # 天气查询工具（Open-Meteo，中文城市名）
│   ├── reranker.py                 # 检索重排器（交叉编码器 + 轻量降级）
│   ├── auth.py                     # JWT + API Key 认证
│   ├── users.py                    # 用户管理（PBKDF2 密码哈希）
│   ├── user_scope.py               # 用户集合隔离（个人集合 + 归属判断）
│   ├── conversations/
│   │   └── manager.py              # 对话管理（基于 PostgreSQL）
│   ├── cache/
│   │   └── __init__.py             # LRU 问答缓存（按来源文件打标签精准失效）
│   ├── ratelimit/                  # 🚦 分级限流（IP/全局/LLM 令牌桶 + 降级）
│   │   ├── middleware.py
│   │   ├── sliding_window.py
│   │   ├── token_bucket.py
│   │   ├── degradation.py
│   │   └── redis_client.py         # Redis 连接（不可用自动降级内存）
│   ├── storage.py                  # 文件存储抽象（local | oss）
│   ├── ingest_tasks.py             # 异步大文件导入任务管理器
│   └── utils/
│       └── logger.py               # 统一日志配置
│
├── data/
│   ├── documents/                  # 📂 存放待导入的原始文档
│   ├── processed/                  # 💾 处理后的文本缓存
│   └── chroma_db/                  # 🗃️ ChromaDB 持久化数据（chroma 后端时）
│
├── frontend/                       # 🌐 React 前端界面
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js              # Vite 构建配置（含 API 代理）
│   ├── dist/                       # 构建产物（生产模式使用）
│   └── src/
│       ├── main.jsx
│       ├── App.jsx                 # 主组件（认证 + 状态管理 + 布局）
│       ├── App.css                 # 全局样式（亮/暗主题）
│       ├── services/
│       │   └── api.js              # API 封装层（31 个接口 + SSE）
│       └── components/
│           ├── LoginPage.jsx       # 登录页
│           ├── Sidebar.jsx         # 侧边栏（统计/上传/集合/操作）
│           ├── ChatArea.jsx        # 对话区域（流式渲染）
│           ├── MessageBubble.jsx   # 消息气泡（来源引用 + 相关问题）
│           ├── ConversationPanel.jsx  # 会话管理面板
│           ├── CollectionModal.jsx    # 集合管理弹窗
│           ├── UploadQueue.jsx        # 上传队列（异步进度）
│           ├── UserManagementModal.jsx  # 用户管理弹窗
│           └── ChangePasswordModal.jsx  # 修改密码弹窗
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

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 BAILIAN_API_KEY，如使用 pgvector 还需配置 DATABASE_URL

# 5. 将企业文档放入 data/documents/ 目录

# 6. 导入文档到知识库
python ingest.py -d data/documents/ -r

# 7. 启动 API 服务
python app.py

# 8. 打开另一个终端，测试问答
python query.py -i
# 或访问 http://localhost:8000 使用 Web 界面（admin / admin123）
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

**原因**：知识库为空，或文档未成功导入，或检索分数低于相关性阈值。

**解决方法**：
```bash
# 检查知识库状态
python ingest.py --stats

# 如果 total_chunks 为 0，需要导入文档
python ingest.py -d data/documents/ -r
```

> 系统会在知识库为空时自动使用 LLM 通用知识回答，并在末尾提醒上传文档；检索不到相关内容时回退通用知识（`answer_type: general`）。

### Q: 提示 "认证失败" 或 401

**原因**：`AUTH_ENABLED=true` 时未携带令牌，或令牌过期。

**解决方法**：
1. 通过 `/api/auth/login` 重新获取 JWT 令牌
2. 请求头携带 `Authorization: Bearer <token>`（或配置的 `X-API-Key`）
3. 开发环境可设置 `AUTH_ENABLED=false` 关闭认证

### Q: 上传文件提示 "文件过大" 或 422

**原因**：超过 100MB 上限，或扩展名不在支持列表，或解析失败（加密/损坏）。

**解决方法**：
1. 大文件压缩或拆分后上传（>10MB 会自动走异步导入）
2. 扫描版 PDF / 图片需安装 OCR：`pip install paddlepaddle paddleocr`
3. 加密文件请先解密

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
RETRIEVAL_CANDIDATE_K=20  # 重排前召回更多候选
RERANK_ENABLED=true    # 开启重排（交叉编码器或轻量降级）
```

### Q: 数据库相关错误（psycopg2 / asyncpg / pgvector）

**原因**：使用 `VECTOR_STORE_TYPE=pg` 但 PostgreSQL 未就绪或未安装 pgvector 扩展。

**解决方法**：
1. 确认 PostgreSQL 已启动，`DATABASE_URL` 正确
2. 执行 `CREATE EXTENSION IF NOT EXISTS vector;`
3. 或切换回零配置的 ChromaDB：`VECTOR_STORE_TYPE=chroma`

### Q: 端口被占用怎么办？

```bash
# 使用其他端口启动
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Q: 问答非常频繁被限流（429）

**原因**：触发了分级限流（每 IP 10 QPS / 全局 50 QPS / LLM 8 QPS）。

**解决方法**：在 `.env` 中调高阈值，或减少并发：

```ini
RATE_LIMIT_IP_QPS=20
RATE_LIMIT_GLOBAL_QPS=100
RATE_LIMIT_LLM_QPS=15
```

---

## 📜 许可证

本项目仅供学习和研究使用。

---

## 🙏 致谢

- [阿里云百炼 (DashScope)](https://bailian.console.aliyun.com/) — 提供通义千问大模型、文本嵌入与重排服务
- [PostgreSQL + pgvector](https://github.com/pgvector/pgvector) — 高性能向量数据库与统一存储
- [ChromaDB](https://www.trychroma.com/) — 轻量嵌入式向量数据库
- [FastAPI](https://fastapi.tiangolo.com/) — 现代化 Python Web 框架
- [React](https://react.dev/) — 前端 UI 框架

---

> **提示**：如果在使用过程中遇到任何问题，请查看 `logs/app.log` 日志文件获取详细错误信息。
