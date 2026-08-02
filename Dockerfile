# ============================================================================
# 企业知识库问答系统 - Dockerfile
#
# 多阶段构建:
#   Stage 1 (frontend-builder) — Node.js 构建 React 前端
#   Stage 2 (app)              — Python 3.12-slim 运行 FastAPI 后端
#
# 构建:   docker build -t enterprise-kb .
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: 构建前端
# ---------------------------------------------------------------------------
FROM node:22-alpine AS frontend-builder

WORKDIR /build

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: 运行后端
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 系统依赖（PyMuPDF 等需要）
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
         gcc libpq-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 复制 Python 代码
COPY . .

# 复制前端构建产物（生产模式由 FastAPI 托管）
COPY --from=frontend-builder /build/dist /app/frontend/dist

# 创建必要目录
RUN mkdir -p /app/data/chroma_db \
             /app/data/documents \
             /app/data/processed \
             /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

ENTRYPOINT ["python", "app.py"]
