import React, { useState, useRef } from 'react'

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

export default function MessageBubble({ message }) {
  const [showSources, setShowSources] = useState(false)
  const [highlightedIndex, setHighlightedIndex] = useState(null)
  const sourceRefs = useRef({})
  const isUser = message.role === 'user'
  const isStreaming = message.streaming
  const related = message.related_questions || []

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
