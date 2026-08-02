import React, { useState, useRef, useCallback, useEffect } from 'react'
import {
  deleteCollection, createCollection, renameCollection,
} from '../services/api'
import CollectionModal from './CollectionModal'
import ConversationPanel from './ConversationPanel'
import UploadQueue from './UploadQueue'

export default function Sidebar({
  collections,
  conversations,
  currentConvId,
  serverOnline,
  onUploadSuccess,
  onClearMessages,
  onRefresh,
  onSwitchConversation,
  onCreateConversation,
  onConversationsChange,
}) {
  const [viewCollection, setViewCollection] = useState(null)
  const [toast, setToast] = useState(null)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [createValue, setCreateValue] = useState('')
  const [showRenameCollModal, setShowRenameCollModal] = useState(false)
  const [renameCollTarget, setRenameCollTarget] = useState(null)
  const [renameCollValue, setRenameCollValue] = useState('')
  const [uploadCollection, setUploadCollection] = useState('knowledge_base')

  // 集合名列表（兼容新旧格式）
  const collectionNames = collections.map(c => typeof c === 'string' ? c : (c.name || c))

  // 同步 uploadCollection：如果当前值不在集合列表中，切换到第一个可用集合
  useEffect(() => {
    if (collectionNames.length > 0 && !collectionNames.includes(uploadCollection)) {
      setUploadCollection(collectionNames[0])
    }
  }, [collectionNames, uploadCollection])

  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 3000)
  }, [])

  // 集合详情（兼容新旧格式）
  const collectionDetails = collections.map(c => typeof c === 'string' ? { name: c, chunk_count: 0 } : c)

  // 创建集合
  const handleCreateCollection = useCallback(async () => {
    if (!createValue.trim()) return
    try {
      await createCollection(createValue.trim())
      showToast(`集合 "${createValue.trim()}" 已创建`, 'success')
      setShowCreateModal(false)
      setCreateValue('')
      onRefresh()
    } catch (err) {
      showToast(err.message || '创建失败', 'error')
    }
  }, [createValue, onRefresh, showToast])

  // 重命名集合
  const handleRenameCollection = useCallback(async () => {
    if (!renameCollValue.trim() || !renameCollTarget) return
    try {
      await renameCollection(renameCollTarget, renameCollValue.trim())
      showToast(`集合已重命名为 "${renameCollValue.trim()}"`, 'success')
      setShowRenameCollModal(false)
      setRenameCollTarget(null)
      setRenameCollValue('')
      onRefresh()
    } catch (err) {
      showToast(err.message || '重命名失败', 'error')
    }
  }, [renameCollTarget, renameCollValue, onRefresh, showToast])

  // 删除集合
  const handleDeleteCollection = useCallback(async () => {
    if (!deleteTarget) return
    try {
      await deleteCollection(deleteTarget)
      // 如果删除的是当前上传目标集合，切回默认
      if (uploadCollection === deleteTarget) setUploadCollection('knowledge_base')
      showToast(`集合 "${deleteTarget}" 已删除`, 'success')
      setShowDeleteModal(false)
      setDeleteTarget(null)
      onRefresh()
    } catch (err) {
      showToast(err.message || '删除失败', 'error')
    }
  }, [deleteTarget, uploadCollection, onRefresh, showToast])

  return (
    <aside className="sidebar">
      {/* 对话历史 */}
      <div className="sidebar-section">
        <ConversationPanel
          conversations={conversations}
          currentConvId={currentConvId}
          serverOnline={serverOnline}
          onSwitch={onSwitchConversation}
          onCreate={onCreateConversation}
          onConversationsChange={onConversationsChange}
        />
      </div>

      {/* 集合管理 */}
      <div className="sidebar-section">
        <div className="coll-header">
          <span className="sidebar-section-title">集合列表</span>
          <button className="btn-conv-new" onClick={() => setShowCreateModal(true)} title="新建集合" disabled={!serverOnline}>＋</button>
        </div>

        {collectionNames.length > 0 ? (
          <div className="collection-list">
            {collectionDetails.map((item) => {
              const name = item.name || item
              const count = item.chunk_count || 0
              const isActive = uploadCollection === name
              return (
                <div key={name} className={`collection-item ${isActive ? 'active' : ''}`}>
                  <div className="collection-item-main" onClick={() => setUploadCollection(name)} title="点击切换">
                    <span className="collection-icon">📦</span>
                    <span className="collection-name">{name}</span>
                    <span className="chunk-group-count">{count}</span>
                  </div>
                  <div className="collection-item-actions">
                    <button className="coll-btn-sm" onClick={() => { setViewCollection(name) }} title="查看内容">📄</button>
                    <button className="coll-btn-sm" onClick={() => { setRenameCollTarget(name); setRenameCollValue(name); setShowRenameCollModal(true) }} title="重命名">✏️</button>
                    <button className="coll-btn-sm coll-btn-del" onClick={() => { setDeleteTarget(name); setShowDeleteModal(true) }} title="删除">🗑️</button>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (<div className="empty-text">暂无集合，点击 ＋ 新建</div>)}
      </div>

      {/* 上传文档 */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">上传文档</div>
        {collectionNames.length > 0 && (
          <div className="upload-collection-selector">
            <span className="upload-collection-label">上传到：</span>
            <select
              className="upload-collection-select"
              value={uploadCollection}
              onChange={(e) => setUploadCollection(e.target.value)}
            >
              {collectionNames.map(n => <option key={n} value={n}>{n} ({collectionDetails.find(c => (c.name || c) === n)?.chunk_count || 0} 块)</option>)}
            </select>
          </div>
        )}
        <UploadQueue
          uploadCollection={uploadCollection}
          onUploadSuccess={onUploadSuccess}
          serverOnline={serverOnline}
        />
      </div>

      {/* 操作按钮 */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">操作</div>
        <div className="sidebar-actions">
          <button className="btn btn-sm" onClick={onClearMessages} disabled={!serverOnline}>🗑️ 清空对话</button>
          <button className="btn btn-sm" onClick={onRefresh} disabled={!serverOnline}>🔄 刷新信息</button>
        </div>
      </div>

      {/* 创建集合弹窗 */}
      {showCreateModal && (
        <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 400 }}>
            <div className="modal-title" style={{ padding: '20px 24px 0' }}>📦 新建集合</div>
            <div className="modal-body" style={{ padding: '12px 24px 20px' }}>
              <p style={{ marginBottom: 12, fontSize: 14, color: 'var(--color-text-secondary)' }}>输入新集合的名称：</p>
              <input className="rename-input" value={createValue} onChange={(e) => setCreateValue(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') handleCreateCollection() }} placeholder="如：人事制度、产品文档" autoFocus />
            </div>
            <div className="modal-actions" style={{ padding: '12px 24px', borderTop: '1px solid var(--color-border)' }}>
              <button className="btn btn-sm" onClick={() => setShowCreateModal(false)}>取消</button>
              <button className="btn btn-sm btn-primary" onClick={handleCreateCollection} disabled={!createValue.trim()}>创建</button>
            </div>
          </div>
        </div>
      )}

      {/* 重命名集合弹窗 */}
      {showRenameCollModal && (
        <div className="modal-overlay" onClick={() => setShowRenameCollModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 400 }}>
            <div className="modal-title" style={{ padding: '20px 24px 0' }}>✏️ 重命名集合</div>
            <div className="modal-body" style={{ padding: '12px 24px 20px' }}>
              <p style={{ marginBottom: 12, fontSize: 14, color: 'var(--color-text-secondary)' }}>为 "{renameCollTarget}" 输入新名称：</p>
              <input className="rename-input" value={renameCollValue} onChange={(e) => setRenameCollValue(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') handleRenameCollection() }} autoFocus />
            </div>
            <div className="modal-actions" style={{ padding: '12px 24px', borderTop: '1px solid var(--color-border)' }}>
              <button className="btn btn-sm" onClick={() => setShowRenameCollModal(false)}>取消</button>
              <button className="btn btn-sm btn-primary" onClick={handleRenameCollection} disabled={!renameCollValue.trim()}>确认</button>
            </div>
          </div>
        </div>
      )}

      {/* 删除集合弹窗 */}
      {showDeleteModal && (
        <div className="modal-overlay" onClick={() => setShowDeleteModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-title">⚠️ 确认删除</div>
            <div className="modal-body">
              确定要删除集合 "<strong>{deleteTarget}</strong>" 吗？<br />
              该操作会同时删除集合内的所有文档块和 OSS 上的原始文件，不可恢复。
            </div>
            <div className="modal-actions">
              <button className="btn btn-sm" onClick={() => setShowDeleteModal(false)}>取消</button>
              <button className="btn btn-sm btn-danger" onClick={handleDeleteCollection}>确认删除</button>
            </div>
          </div>
        </div>
      )}

      {/* 集合内容查看弹窗 */}
      {viewCollection && (
        <CollectionModal collectionName={viewCollection} onClose={() => setViewCollection(null)} />
      )}
    </aside>
  )
}