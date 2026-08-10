import React, { useState, useEffect, useCallback, useRef } from 'react'
import { getTraces, getTraceDetail } from '../services/api'

// 环节名称映射为友好中文标签
const STAGE_LABEL = {
  input: '🔵 输入',
  rewrite: '✏️ 问题重写',
  force_search: '🔎 强制预检索',
  retrieval: '📚 召回',
  rerank: '⚖️ 重排',
  prompt: '📄 提示词',
  tool_mode: '🛠️ 工具模式',
  tool_result: '🛠️ 工具结果',
  generation: '💬 生成',
  cache_hit: '⚡ 缓存命中',
}

function formatTime(ts) {
  if (!ts) return '-'
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString('zh-CN', { hour12: false }) + '.' + String(d.getMilliseconds()).padStart(3, '0')
}

// 渲染一个环节的输入/输出内容（JSON 友好的折叠展示）
function StageView({ stage }) {
  const [open, setOpen] = useState(true)
  const label = STAGE_LABEL[stage.stage] || stage.stage

  return (
    <div className="trace-stage">
      <div className="trace-stage-header" onClick={() => setOpen(!open)}>
        <span className="trace-stage-label">{label}</span>
        <span className="trace-stage-time">{formatTime(stage.ts)}</span>
        <span className="trace-stage-toggle">{open ? '▼' : '▶'}</span>
      </div>
      {open && (
        <div className="trace-stage-body">
          {stage.input !== null && stage.input !== undefined && (
            <div className="trace-stage-io">
              <span className="trace-io-label">输入</span>
              <pre className="trace-io-content">{typeof stage.input === 'string' ? stage.input : JSON.stringify(stage.input, null, 2)}</pre>
            </div>
          )}
          {stage.output !== null && stage.output !== undefined && (
            <div className="trace-stage-io">
              <span className="trace-io-label">输出</span>
              <pre className="trace-io-content">{typeof stage.output === 'string' ? stage.output : JSON.stringify(stage.output, null, 2)}</pre>
            </div>
          )}
          {stage.input == null && stage.output == null && (
            <div className="trace-stage-empty">（无内容）</div>
          )}
        </div>
      )}
    </div>
  )
}

export default function TraceModal({ onClose }) {
  const [traces, setTraces] = useState([])
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const timerRef = useRef(null)

  const loadTraces = useCallback(async () => {
    try {
      const res = await getTraces({ limit: 50 })
      setTraces(res.data || [])
      setError(null)
    } catch (err) {
      setError(err.message || '加载链路追踪失败')
    } finally {
      setLoading(false)
    }
  }, [])

  const openDetail = useCallback(async (traceId) => {
    try {
      const res = await getTraceDetail(traceId)
      setDetail(res.data || null)
      setError(null)
    } catch (err) {
      setError(err.message || '加载链路详情失败')
    }
  }, [])

  // 初始加载 + 每 3 秒轮询列表（实时查看）
  useEffect(() => {
    loadTraces()
    timerRef.current = setInterval(loadTraces, 3000)
    return () => clearInterval(timerRef.current)
  }, [loadTraces])

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content trace-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-left">
            <span className="modal-icon">🔍</span>
            <div>
              <div className="modal-title">查询链路追踪</div>
              <div className="modal-subtitle">实时查看每次问答各环节的输入/输出（每 3 秒刷新）</div>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          {error && <div className="modal-error">⚠️ {error}</div>}

          {detail ? (
            <div className="trace-detail">
              <button className="btn btn-sm btn-ghost trace-back-btn" onClick={() => setDetail(null)}>
                ← 返回列表
              </button>
              <div className="trace-detail-meta">
                <span className="trace-detail-question">❓ {detail.question}</span>
                <span className="trace-detail-type">类型: {detail.answer_type || '-'}</span>
                <span className="trace-detail-dur">耗时: {detail.duration_ms != null ? `${detail.duration_ms}ms` : '-'}</span>
                <span className="trace-detail-id">ID: {detail.trace_id}</span>
              </div>
              <div className="trace-stages">
                {detail.stages?.map((s, i) => (
                  <StageView key={i} stage={s} />
                ))}
              </div>
            </div>
          ) : loading ? (
            <div className="modal-loading"><p>加载中...</p></div>
          ) : traces.length === 0 ? (
            <div className="modal-empty">
              <p>暂无追踪记录</p>
              <p className="trace-empty-hint">发起一次问答后，这里会显示完整的链路追踪</p>
            </div>
          ) : (
            <div className="trace-list">
              {traces.map((t) => (
                <div key={t.trace_id} className="trace-item" onClick={() => openDetail(t.trace_id)}>
                  <div className="trace-item-q">❓ {t.question || '-'}</div>
                  <div className="trace-item-meta">
                    <span className={`trace-badge ${t.answer_type || ''}`}>{t.answer_type || '-'}</span>
                    <span className="trace-item-stages">{t.stage_count || 0} 环节</span>
                    <span className="trace-item-dur">{t.duration_ms != null ? `${t.duration_ms}ms` : '-'}</span>
                    <span className="trace-item-time">{formatTime(t.started_at)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn btn-sm" onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  )
}
