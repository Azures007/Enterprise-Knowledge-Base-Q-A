import React, { useState, useRef } from 'react'
import { sendMessageFeedback } from '../services/api'

// 渲染回答内容，把 [N] 引用标注渲染为可点击的高亮编号
function AnswerContent({ content, onCiteClick }) {
  if (!content) return null
  // 匹配 [1]、[2]、[1][2] 形式的引用标注
  const parts = content.split(/(\[\d+\])/g)
  return (
    <>
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/)
        if (m) {
          const n = parseInt(m[1], 10)
          return (
            <button
              key={i}
              className="citation-link"
              onClick={() => onCiteClick(n)}
              title={`查看来源 [${n}]`}
            >
              [{n}]
            </button>
          )
        }
        return <span key={i}>{part}</span>
      })}
    </>
  )
}

// 单个来源项：显示文件名 + 分数，可展开查看被检索到的分块
function SourceItem({ src, highlighted, onChunkClick }) {
  const [showChunks, setShowChunks] = useState(false)
  const chunks = src.chunks || []

  return (
    <div className={`source-item ${highlighted ? 'source-item-highlighted' : ''}`}>
      <div className="source-item-header">
        {src.index && <span className="source-index">[{src.index}]</span>}
        <span className="source-score">{(src.score * 100).toFixed(1)}%</span>
        <span className="source-name">{src.filename || '未知来源'}</span>
        {src.page && (
          <span style={{ color: 'var(--color-text-muted)' }}>p.{src.page}</span>
        )}
        {chunks.length > 0 && (
          <button
            className="source-chunks-toggle"
            onClick={() => setShowChunks(!showChunks)}
          >
            {showChunks ? '收起分块 ▲' : `查看 ${chunks.length} 个分块 ▼`}
          </button>
        )}
      </div>

      {showChunks && (
        <div className="source-chunks">
          {chunks.map((chunk, i) => (
            <div key={i} className="source-chunk" onClick={() => onChunkClick?.(src, chunk)}>
              <div className="source-chunk-meta">
                <span>分块 {chunk.chunk_index !== undefined && chunk.chunk_index !== null
                  ? chunk.chunk_index + 1
                  : i + 1}</span>
                <span className="source-chunk-score">
                  {(chunk.score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="source-chunk-content">{chunk.content}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function MessageBubble({ message, currentConvId }) {
  const [showSources, setShowSources] = useState(false)
  const [highlightedIndex, setHighlightedIndex] = useState(null)
  const sourceRefs = useRef({})
  const isUser = message.role === 'user'
  const isStreaming = message.streaming
  const related = message.related_questions || []

  // 反馈状态：初始来自历史消息字段（message.feedback: 1=赞 -1=踩 null=无）
  const [feedback, setFeedback] = useState(message.feedback ?? null)
  const [feedbackComment, setFeedbackComment] = useState('')
  const [showFeedbackInput, setShowFeedbackInput] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  // 点击引用 [N] → 高亮对应来源并滚动到它
  const handleCiteClick = (n) => {
    setShowSources(true)
    setHighlightedIndex(n)
    // 滚动到对应来源项
    setTimeout(() => {
      const el = sourceRefs.current[n]
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 100)
  }

  const handleRelatedClick = (q) => {
    // 把相关问题作为新问题发送（由 ChatArea 通过 prop 处理）
    if (window.__askQuestion) window.__askQuestion(q)
  }

  // 提交反馈（1=赞 / -1=踩 / 0=清除）；comment 仅点踩时可填
  const submitFeedback = async (value, comment) => {
    if (submitting || !message.id || currentConvId == null) return
    // 再点相同按钮 → 清除
    const target = (value === feedback) ? 0 : value
    const finalComment = (target === -1 && comment) ? comment : null
    setSubmitting(true)
    try {
      await sendMessageFeedback(currentConvId, message.id, {
        feedback: target,
        comment: finalComment,
      })
      setFeedback(target === 0 ? null : target)
      if (target !== -1) {
        setShowFeedbackInput(false)
        setFeedbackComment('')
      } else {
        setShowFeedbackInput(false)
      }
    } catch (err) {
      console.warn('反馈提交失败:', err.message)
    } finally {
      setSubmitting(false)
    }
  }

  const handleFeedbackClick = (value) => {
    if (value === -1 && feedback !== -1) {
      // 点踩：先展开原因输入框
      setFeedbackComment('')
      setShowFeedbackInput(true)
      return
    }
    submitFeedback(value)
  }

  return (
    <div className={`message ${isUser ? 'user' : 'ai'}`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-body">
        <div
          className={`message-content ${isStreaming ? 'streaming-cursor' : ''}`}
        >
          <AnswerContent content={message.content} onCiteClick={handleCiteClick} />
          {isStreaming && !message.content && (
            <div className="typing-indicator">
              <span /><span /><span />
            </div>
          )}
        </div>

        {!isUser && message.is_stale && (
          <div className="message-stale-hint" title="引用的文档可能已更新或删除">
            ⚠️ 基于已更新的知识库，此回答可能已过时
          </div>
        )}

        {/* 用户反馈：点赞 / 点踩 */}
        {!isUser && !isStreaming && message.id && (
          <div className="feedback-actions">
            <button
              className={`feedback-btn ${feedback === 1 ? 'active-up' : ''}`}
              onClick={() => handleFeedbackClick(1)}
              title="这个回答有帮助"
              disabled={submitting}
            >
              👍 {feedback === 1 ? '已点赞' : '有用'}
            </button>
            <button
              className={`feedback-btn ${feedback === -1 ? 'active-down' : ''}`}
              onClick={() => handleFeedbackClick(-1)}
              title="这个回答需要改进"
              disabled={submitting}
            >
              👎 {feedback === -1 ? '已点踩' : '需改进'}
            </button>
            {feedback && (
              <button
                className="feedback-clear"
                onClick={() => submitFeedback(0)}
                title="清除反馈"
                disabled={submitting}
              >
                撤销
              </button>
            )}

            {showFeedbackInput && (
              <div className="feedback-reason">
                <input
                  type="text"
                  value={feedbackComment}
                  onChange={(e) => setFeedbackComment(e.target.value)}
                  placeholder="选填：告诉我们哪里不对（可选）"
                  maxLength={500}
                  autoFocus
                />
                <button
                  className="btn btn-sm"
                  onClick={() => submitFeedback(-1, feedbackComment)}
                  disabled={submitting}
                >
                  提交
                </button>
                <button
                  className="btn btn-sm btn-ghost"
                  onClick={() => setShowFeedbackInput(false)}
                >
                  取消
                </button>
              </div>
            )}
          </div>
        )}

        {/* 相关问题推荐 */}
        {!isUser && !isStreaming && related.length > 0 && (
          <div className="related-questions">
            <div className="related-questions-title">💡 相关问题</div>
            {related.map((q, i) => (
              <button key={i} className="related-question-btn" onClick={() => handleRelatedClick(q)}>
                {q}
              </button>
            ))}
          </div>
        )}

        {/* 来源引用 */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="sources-section">
            <div
              className="sources-toggle"
              onClick={() => setShowSources(!showSources)}
            >
              📚 {showSources ? '收起来源' : `查看 ${message.sources.length} 个来源`}
              <span>{showSources ? '▲' : '▼'}</span>
            </div>
            {showSources && (
              <div className="sources-list">
                {message.sources.map((src, i) => (
                  <div key={i} ref={(el) => { if (src.index) sourceRefs.current[src.index] = el }}>
                    <SourceItem
                      src={src}
                      highlighted={highlightedIndex === src.index}
                      onChunkClick={() => {}}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
