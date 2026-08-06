#!/usr/bin/env bash
# =============================================================================
# 企业知识库问答系统 - 一键启动脚本（后端，生产模式）
#
# 功能：
#   1. 清理 8000 端口残留进程（避免多个实例抢端口/资源导致段错误）
#   2. 以单实例方式启动后端（FastAPI 托管前端 dist）
#   3. 可选：--dev 模式启动前端开发服务器
#
# 用法：
#   ./start.sh            # 生产模式（后端 8000，托管已构建的前端）
#   ./start.sh --dev      # 开发模式（后端 8000 + 前端 5173 热重载）
#   ./start.sh --stop     # 停止后端（按 app.py 命令行匹配进程）
# =============================================================================
set -e

PORT=8000
FRONTEND_PORT=5173
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 颜色输出
GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'

log()  { echo -e "${GREEN}[start]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
err()  { echo -e "${RED}[error]${NC} $1"; }

# ---------------------------------------------------------------------------
# 停止所有本项目的 app.py 实例（按命令行精确匹配，避免误杀其他 python）
# ---------------------------------------------------------------------------
stop_backend() {
    warn "正在停止现有后端实例..."
    # Windows: taskkill 按命令行匹配 app.py
    for pid in $(wmic process where "name='python.exe'" get ProcessId,CommandLine 2>/dev/null \
                 | grep -i "app.py" | awk '{print $NF}' | grep -E '^[0-9]+$'); do
        taskkill //PID "$pid" //F 2>/dev/null && log "已停止后端进程 PID=$pid"
    done
    # 兜底：若仍有进程占用端口
    local listener
    listener=$(netstat -ano 2>/dev/null | grep ":$PORT " | grep LISTENING | awk '{print $NF}' | head -1)
    if [ -n "$listener" ]; then
        taskkill //PID "$listener" //F 2>/dev/null && log "已清理端口 $PORT 占用进程 PID=$listener"
    fi
    sleep 2
    # 验证端口已释放
    if netstat -ano 2>/dev/null | grep ":$PORT " | grep -q LISTENING; then
        err "端口 $PORT 仍被占用，请手动检查"
        exit 1
    fi
    log "端口 $PORT 已释放"
}

# ---------------------------------------------------------------------------
# 检查依赖
# ---------------------------------------------------------------------------
check_env() {
    if [ ! -f ".env" ]; then
        err "缺少 .env 配置文件，请先: cp .env.example .env 并填写 API Key"
        exit 1
    fi
    if [ ! -f "frontend/dist/index.html" ]; then
        warn "未找到前端构建产物 frontend/dist，先构建前端..."
        (cd frontend && npm install && npm run build)
    fi
}

# ---------------------------------------------------------------------------
# 启动后端（单实例）
# ---------------------------------------------------------------------------
start_backend() {
    check_env
    log "启动后端 (http://localhost:$PORT) ..."
    python app.py
}

# ---------------------------------------------------------------------------
# 开发模式：后端 + 前端 dev server
# ---------------------------------------------------------------------------
start_dev() {
    check_env
    log "启动后端 (http://localhost:$PORT) ..."
    python app.py &
    local backend_pid=$!
    log "后端已启动 (PID=$backend_pid)，等待就绪..."
    # 等待后端健康检查通过（最多 60 秒）
    for i in $(seq 1 60); do
        if curl -s --max-time 2 "http://localhost:$PORT/api/health" | grep -q '"status":"ok"'; then
            log "后端就绪 (${i} 秒)"
            break
        fi
        sleep 1
    done
    log "启动前端开发服务器 (http://localhost:$FRONTEND_PORT) ..."
    (cd frontend && npm run dev)
}

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
case "${1:-}" in
    --stop)
        stop_backend
        ;;
    --dev)
        stop_backend
        start_dev
        ;;
    *)
        stop_backend
        start_backend
        ;;
esac
