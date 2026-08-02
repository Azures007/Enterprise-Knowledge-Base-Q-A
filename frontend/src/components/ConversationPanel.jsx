import React, { useState, useCallback } from 'react'
import {
  createConversation,
  deleteConversation,
  batchDeleteConversations,
  updateConversationTitle,
} from '../services/api'

export default function ConversationPanel({
  conversations,
  currentConvId,
  onSwitch,
  onConversationsChange,
  serverOnline,
}) {
  const [editingId, setEditingId] = useState(null)
  const [editTitle, setEditTitle] = useState('')
  const [selectMode, setSelectMode] = useState(false)
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [deleting, setDeleting] = useState(false)

  // 新建对话
  const handleCreate = useCallback(async () => {
    try {
      const res = await createConversation()
      const newConv = res.data
      onConversationsChange((prev) => [newConv, ...prev])
      onSwitch(newConv.id)
    } catch (err) {
      console.error('创建对话失败:', err)
    }
  }, [onSwitch, onConversationsChange])

  // 删除单个对话
  const handleDelete = useCallback(
    async (e, convId) => {
      e.stopPropagation()
      try {
        await deleteConversation(convId)
        onConversationsChange((prev) => prev.filter((c) => c.id !== convId))
        if (currentConvId === convId) {
          onSwitch(null)
        }
      } catch (err) {
        console.error('删除对话失败:', err)
      }
    },
    [currentConvId, onSwitch, onConversationsChange],
  )

  // 批量删除
  const handleBatchDelete = useCallback(async () => {
    if (selectedIds.size === 0) return
    if (!confirm(`确定要删除选中的 ${selectedIds.size} 个对话吗？`)) return
    setDeleting(true)
    try {
      const ids = Array.from(selectedIds)
      await batchDeleteConversations(ids)
      onConversationsChange((prev) => prev.filter((c) => !selectedIds.has(c.id)))
      if (selectedIds.has(currentConvId)) {
        onSwitch(null)
      }
      setSelectedIds(new Set())
      setSelectMode(false)
    } catch (err) {
      console.error('批量删除失败:', err)
    } finally {
      setDeleting(false)
    }
  }, [selectedIds, currentConvId, onSwitch, onConversationsChange])

  // 切换选择/取消
  const toggleSelect = useCallback((convId) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(convId)) {
        next.delete(convId)
      } else {
        next.add(convId)
      }
      return next
    })
  }, [])

  // 全选/取消全选
  const toggleSelectAll = useCallback(() => {
    if (selectedIds.size === conversations.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(conversations.map((c) => c.id)))
    }
  }, [conversations, selectedIds])

  // 开始编辑标题
  const startEdit = useCallback((e, conv) => {
    e.stopPropagation()
    setEditingId(conv.id)
    setEditTitle(conv.title)
  }, [])

  // 保存标题
  const saveTitle = useCallback(
    async (convId) => {
      const title = editTitle.trim()
      if (!title) {
        setEditingId(null)
        return
      }
      try {
        await updateConversationTitle(convId, title)
        onConversationsChange((prev) =>
          prev.map((c) => (c.id === convId ? { ...c, title } : c)),
        )
      } catch (err) {
        console.error('修改标题失败:', err)
      }
      setEditingId(null)
    },
    [editTitle, onConversationsChange],
  )

  return (
    <div className="conv-panel">
      <div className="conv-header">
        <span className="sidebar-section-title">对话历史</span>
        <div className="conv-header-actions">
          {selectMode ? (
            <>
              <button
                className="btn-conv-sm"
                onClick={toggleSelectAll}
                title={selectedIds.size === conversations.length ? '取消全选' : '全选'}
              >
                {selectedIds.size === conversations.length ? '取消全选' : '全选'}
              </button>
              <button
                className="btn-conv-sm btn-conv-del"
                onClick={handleBatchDelete}
                disabled={selectedIds.size === 0 || deleting}
              >
                {deleting ? '删除中...' : `删除 (${selectedIds.size})`}
              </button>
              <button className="btn-conv-sm" onClick={() => { setSelectMode(false); setSelectedIds(new Set()) }}>
                完成
              </button>
            </>
          ) : (
            <>
              <button
                className="btn-conv-sm"
                onClick={() => setSelectMode(true)}
                disabled={conversations.length === 0}
                title="批量删除"
              >
                批量
              </button>
              <button
                className="btn-conv-new"
                onClick={handleCreate}
                disabled={!serverOnline}
                title="新建对话"
              >
                ＋
              </button>
            </>
          )}
        </div>
      </div>

      <div className="conv-list">
        {conversations.length === 0 && (
          <div className="conv-empty">暂无对话，点击 ＋ 新建</div>
        )}

        {conversations.map((conv) => (
          <div
            key={conv.id}
            className={`conv-item ${currentConvId === conv.id ? 'active' : ''} ${selectMode && selectedIds.has(conv.id) ? 'selected' : ''}`}
            onClick={() => selectMode ? toggleSelect(conv.id) : onSwitch(conv.id)}
          >
            {selectMode && (
              <div className="conv-checkbox">
                {selectedIds.has(conv.id) ? '☑' : '☐'}
              </div>
            )}
            <div className="conv-item-icon">💬</div>

            {editingId === conv.id ? (
              <input
                className="conv-edit-input"
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                onBlur={() => saveTitle(conv.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') saveTitle(conv.id)
                  if (e.key === 'Escape') setEditingId(null)
                }}
                onClick={(e) => e.stopPropagation()}
                autoFocus
              />
            ) : (
              <div
                className="conv-item-title"
                onDoubleClick={(e) => !selectMode && startEdit(e, conv)}
              >
                {conv.title}
              </div>
            )}

            {!selectMode && (
              <div className="conv-item-meta">
                <span className="conv-msg-count">
                  {conv.message_count || 0}
                </span>
                <button
                  className="conv-btn-del"
                  onClick={(e) => handleDelete(e, conv.id)}
                  title="删除对话"
                >
                  ✕
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
