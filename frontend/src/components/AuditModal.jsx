import React, { useState, useEffect, useCallback } from 'react'
import { getAuditQueries, getAuditSummary } from '../services/api'

// 回答类型彩色标签
const TYPE_BADGE = {
  kb: { label: '知识库', className: 'badge-kb' },
  hybrid: { label: '混合', className: 'badge-hybrid' },
  general: { label: '通用', className: 'badge-general' },
}

function formatNumber(n) {
  if (n === null || n === undefined) return '-'
  return Number(n).toLocaleString('zh-CN')
}

function formatLatency(ms) {
  if (ms === null || ms === undefined) return '-'
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`
  return `${ms}ms`
}

export default function AuditModal({ onClose }) {
  const [summary, setSummary] = useState(null)
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(null)
  const [offset, setOffset] = useState(0)
  const LIMIT = 20

  const loadSummary = useCallback(async () => {
    try {
      const res = await getAuditSummary()
      setSummary(res.data || {})
    } catch (err) {
      // 汇总失败不阻断列表
      console.warn('加载审计汇总失败:', err.message)
    }
  }, [])

  const loadRows = useCallback(async (startOffset) => {
    try {
      const res = await getAuditQueries({ limit: LIMIT, offset: startOffset })
      const data = res.data || []
      if (startOffset === 0) {
        setRows(data)
      } else {
        setRows((prev) => [...prev, ...data])
      }
      setOffset(startOffset + data.length)
      setError(null)
    } catch (err) {
      setError(err.message || '加载审计记录失败')
    }
  }, [])

  useEffect(() => {
    ;(async () => {
      setLoading(true)
      await Promise.all([loadSummary(), loadRows(0)])
      setLoading(false)
    })()
  }, [loadSummary, loadRows])

  const handleLoadMore = async () => {
    setLoadingMore(true)
    await loadRows(offset)
    setLoadingMore(false)
  }

  const toggleExpand = (id) => {
    setExpanded((prev) => (prev === id ? null : id))
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content audit-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-left">
            <span className="modal-icon">📊</span>
            <div>
              <div className="modal-title">查询审计</div>
              <div className="modal-subtitle">记录每次知识库问答的详情</div>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          {error && <div className="modal-error">⚠️ {error}</div>}

          {/* 汇总卡片 */}
          <div className="audit-cards">
            <div className="audit-card">
              <div className="audit-card-value">{formatNumber(summary?.total_queries)}</div>
              <div className="audit-card-label">总查询数</div>
            </div>
            <div className="audit-card">
              <div className="audit-card-value">
                {summary?.cache_hit_rate != null ? `${(summary.cache_hit_rate * 100).toFixed(1)}%` : '-'}
              </div>
              <div className="audit-card-label">缓存命中率</div>
            </div>
            <div className="audit-card">
              <div className="audit-card-value">{formatLatency(summary?.avg_latency_ms)}</div>
              <div className="audit-card-label">平均延迟</div>
            </div>
            <div className="audit-card">
              <div className="audit-card-value">{formatNumber(summary?.total_tokens)}</div>
              <div className="audit-card-label">总 Token</div>
            </div>
          </div>

          {/* 热门问题 */}
          {summary?.top_questions?.length > 0 && (
            <div className="audit-top-questions">
              <div className="audit-section-title">🔥 热门问题</div>
              <div className="audit-top-list">
                {summary.top_questions.slice(0, 5).map((q, i) => (
                  <div key={i} className="audit-top-item">
                    <span className="audit-top-rank">{i + 1}</span>
                    <span className="audit-top-question">{q.question}</span>
                    <span className="audit-top-count">{q.count} 次</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 查询记录表格 */}
          <div className="audit-section-title">📋 最近查询</div>
          {loading ? (
            <div className="modal-loading"><p>加载中...</p></div>
          ) : rows.length === 0 ? (
            <div className="modal-empty"><p>暂无查询记录</p></div>
          ) : (
            <div className="audit-table-wrap">
              <table className="audit-table">
                <thead>
                  <tr>
                    <th>时间</th>
                    <th>用户</th>
                    <th>问题</th>
                    <th>类型</th>
                    <th>Token</th>
                    <th>延迟</th>
                    <th>缓存</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const badge = TYPE_BADGE[r.answer_type]
                    return (
                      <React.Fragment key={r.id}>
                        <tr className="audit-row" onClick={() => toggleExpand(r.id)}>
                          <td className="audit-cell-time">
                            {r.created_at ? r.created_at.replace('T', ' ').slice(0, 16) : '-'}
                          </td>
                          <td>{r.username || 'anonymous'}</td>
                          <td className="audit-cell-question" title={r.question}>
                            {r.question ? (r.question.length > 22 ? r.question.slice(0, 22) + '…' : r.question) : '-'}
                          </td>
                          <td>
                            {badge ? (
                              <span className={`audit-badge ${badge.className}`}>{badge.label}</span>
                            ) : (
                              <span className="audit-badge badge-other">{r.answer_type || '-'}</span>
                            )}
                          </td>
                          <td>{formatNumber(r.total_tokens)}</td>
                          <td>{formatLatency(r.latency_ms)}</td>
                          <td>{r.from_cache ? '✅' : '—'}</td>
                        </tr>
                        {expanded === r.id && (
                          <tr className="audit-expand-row">
                            <td colSpan={7}>
                              <div className="audit-expand">
                                <div className="audit-expand-question">
                                  <span className="audit-expand-label">问题</span>
                                  {r.question}
                                </div>
                                {r.answer && (
                                  <div className="audit-expand-answer">
                                    <span className="audit-expand-label">回答</span>
                                    <div className="audit-expand-answer-text">{r.answer}</div>
                                  </div>
                                )}
                                {r.sources && r.sources.length > 0 && (
                                  <div className="audit-expand-sources">
                                    <span className="audit-expand-label">来源</span>
                                    {r.sources.map((s, i) => (
                                      <span key={i} className="audit-expand-source">
                                        📄 {s.filename || s.source || '未知'}
                                      </span>
                                    ))}
                                  </div>
                                )}
                                {r.error_msg && (
                                  <div className="audit-expand-error">❌ {r.error_msg}</div>
                                )}
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          {rows.length > 0 && (
            <button
              className="btn btn-sm"
              onClick={handleLoadMore}
              disabled={loadingMore}
              style={{ marginTop: 12, width: '100%' }}
            >
              {loadingMore ? '加载中...' : '加载更多'}
            </button>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn btn-sm" onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  )
}
