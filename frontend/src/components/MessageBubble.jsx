import React, { useState } from 'react'

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
                  <div key={i} className="source-item">
                    <span className="source-score">
                      {(src.score * 100).toFixed(1)}%
                    </span>
                    <span className="source-name">
                      {src.filename || '未知来源'}
                    </span>
                    {src.page && (
                      <span style={{ color: 'var(--color-text-muted)' }}>
                        p.{src.page}
                      </span>
                    )}
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
