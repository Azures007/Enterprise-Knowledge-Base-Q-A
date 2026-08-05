import React, { useState, useRef, useEffect, useCallback } from 'react'
import MessageBubble from './MessageBubble'
import { streamQuery } from '../services/api'
import {
  addConversationMessage,
  updateMessageContent,
} from '../services/api'

export default function ChatArea({
  messages,
  addMessage,
  updateLastMessage,
  currentConvId,
  serverOnline,
  error,
  onConversationUpdated,
  collections,
  selectedCollection,
  onSelectCollection,
}) {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const abortRef = useRef(null)
  const savedUserMsgId = useRef(null)
  const savedAiMsgId = useRef(null)

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // 自动聚焦输入框
  useEffect(() => {
    if (!loading) inputRef.current?.focus()
  }, [loading])

  // 停止当前回答
  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
  }, [])

  // 发送消息
  const sendMessage = useCallback(async (overrideQuestion) => {
    const question = (overrideQuestion ?? input).trim()
    if (!question || loading || !serverOnline) return

    // 创建 AbortController
    const controller = new AbortController()
    abortRef.current = controller

    setInput('')
    setLoading(true)
    savedAiMsgId.current = null

    // 保存用户消息到后端
    if (currentConvId) {
      try {
        const res = await addConversationMessage(currentConvId, {
          role: 'user',
          content: question,
        })
        savedUserMsgId.current = res.data?.id
        onConversationUpdated?.()
      } catch (err) {
        console.warn('保存用户消息失败:', err.message)
      }
    }

    // 添加用户消息到本地
    addMessage({ role: 'user', content: question })

    // 添加 AI 占位消息
    addMessage({
      role: 'ai',
      content: '',
      sources: null,
      answer_type: null,
      streaming: true,
    })

    let fullAnswer = ''
    let finalSources = []
    let finalAnswerType = 'general'

    streamQuery(
      { question, k: 5, concise: false, collection: selectedCollection },
      {
        onChunk: (chunk) => {
          fullAnswer += chunk
          updateLastMessage((prev) => ({
            ...prev,
            content: fullAnswer,
          }))
        },
        onSources: (sources, answerType) => {
          // 兼容新旧格式：新格式直接传两个参数
        },
        onMeta: (meta) => {
          // 新格式：一次性接收 sources + answer_type
          if (meta.sources) finalSources = meta.sources
          if (meta.answer_type) finalAnswerType = meta.answer_type
          updateLastMessage((prev) => ({
            ...prev,
            sources: meta.sources || prev.sources,
            answer_type: meta.answer_type || prev.answer_type,
          }))
        },
        onDone: async (finalAnswer) => {
          const text = finalAnswer || fullAnswer
          updateLastMessage((prev) => ({
            ...prev,
            content: text,
            sources: finalSources.length > 0 ? finalSources : prev.sources,
            answer_type: finalAnswerType,
            streaming: false,
          }))
          setLoading(false)

          // 保存 AI 消息到后端
          if (currentConvId) {
            try {
              const res = await addConversationMessage(currentConvId, {
                role: 'ai',
                content: text,
                sources: finalSources,
                answer_type: finalAnswerType,
              })
              savedAiMsgId.current = res.data?.id
              // 把后端返回的消息 id 补进本地消息，供反馈按钮关联
              if (res.data?.id) {
                updateLastMessage((prev) => ({ ...prev, id: res.data.id }))
              }
            } catch (err) {
              console.warn('保存 AI 消息失败:', err.message)
            }
          }

          // 刷新对话列表（更新消息数）
          onConversationUpdated?.()
        },
        onError: (errMsg) => {
          updateLastMessage((prev) => ({
            ...prev,
            content: `错误: ${errMsg}`,
            streaming: false,
          }))
          setLoading(false)
        },
        onAbort: () => {
          // 用户手动停止，保留已有内容，追加提示
          updateLastMessage((prev) => ({
            ...prev,
            content: (prev.content || '') + '\n\n--- 用户已停止输出 ---',
            streaming: false,
          }))
          setLoading(false)
          // 保存已收到的部分内容到对话历史
          if (currentConvId && fullAnswer) {
            addConversationMessage(currentConvId, {
              role: 'ai',
              content: fullAnswer + '\n\n--- 用户已停止输出 ---',
              sources: finalSources,
              answer_type: finalAnswerType,
            }).catch(() => {})
          }
          onConversationUpdated?.()
        },
      },
      controller.signal,
    )
  }, [input, loading, serverOnline, currentConvId, addMessage, updateLastMessage, onConversationUpdated])

  // 暴露全局问问题入口（供"相关问题"按钮点击发送）
  // 放在 sendMessage 定义之后，避免 TDZ（暂时性死区）错误
  useEffect(() => {
    window.__askQuestion = (q) => {
      sendMessage(q)
    }
    return () => { delete window.__askQuestion }
  }, [sendMessage])

  // 键盘事件处理
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chat-container">
      {/* 错误提示 */}
      {error && !serverOnline && (
        <div className="error-banner">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* 消息区域 */}
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-icon">🏢</div>
            <h2>欢迎使用企业知识库问答系统</h2>
            <p>
              在下方输入框中描述您的问题，系统将基于知识库中的文档内容为您提供智能回答。
              请先通过左侧面板上传企业文档以初始化知识库。
            </p>
          </div>
        ) : (
          messages.map((msg, i) => (
            <MessageBubble key={msg.id || msg._id || i} message={msg} currentConvId={currentConvId} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 集合选择器 */}
      <div className="chat-collection-bar">
        <span className="chat-collection-label">查询集合：</span>
        <select
          className="chat-collection-select"
          value={selectedCollection || ''}
          onChange={(e) => onSelectCollection(e.target.value || null)}
        >
          <option value="">🌐 自动路由</option>
          {collections.map((c) => {
            const name = typeof c === 'string' ? c : (c.name || c)
            return <option key={name} value={name}>📦 {name}</option>
          })}
        </select>
      </div>

      {/* 输入区域 */}
      <div className="chat-input-container">
        <div className="chat-input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder={
              serverOnline
                ? '请输入您的问题... (Enter 发送, Shift+Enter 换行)'
                : '服务未连接，请等待...'
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={!serverOnline || loading}
          />
          {loading ? (
            <button className="btn-stop" onClick={handleStop} title="停止回答">
              ■ 停止
            </button>
          ) : (
            <button
              className="btn-send"
              onClick={sendMessage}
              disabled={!input.trim() || !serverOnline}
              title="发送"
            >
              ➤
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
