import React, { useState } from 'react'

// 单个来源项：显示文件名 + 分数，可展开查看被检索到的分块
function SourceItem({ src }) {
  const [showChunks, setShowChunks] = useState(false)
  const chunks = src.chunks || []

  return (
    <div className="source-item">
      <div className="source-item-header">
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
            <div key={i} className="source-chunk">
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
  const isUser = message.role === 'user'
  const isStreaming = message.streaming

  return (
    <div className={`message ${isUser ? 'user' : 'ai'}`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-body">
        <div
          className={`message-content ${isStreaming ? 'streaming-cursor' : ''}`}
        >
          {message.content || (isStreaming ? '' : '...')}
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
                  <SourceItem key={i} src={src} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
