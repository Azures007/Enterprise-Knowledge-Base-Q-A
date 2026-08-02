import React, { useState, useEffect, useCallback } from 'react'
import { getCollectionChunks, getCollectionDocuments, deleteDocument } from '../services/api'

export default function CollectionModal({ collectionName, onClose }) {
  const [tab, setTab] = useState('chunks')  // 'chunks' | 'documents'
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [chunks, setChunks] = useState([])
  const [grouped, setGrouped] = useState({})
  const [documents, setDocuments] = useState([])
  const [expandedFiles, setExpandedFiles] = useState({})
  const [expandedChunks, setExpandedChunks] = useState({})
  const [deleting, setDeleting] = useState(null)

  // 加载数据
  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        setLoading(true)
        const [chunksRes, docsRes] = await Promise.all([
          getCollectionChunks(collectionName, 500, 0),
          getCollectionDocuments(collectionName),
        ])
        if (cancelled) return
        setChunks(chunksRes.data.chunks || [])
        setGrouped(chunksRes.data.grouped || {})
        setDocuments(docsRes.data || [])
        const filenames = Object.keys(chunksRes.data.grouped || {})
        if (filenames.length > 0) {
          setExpandedFiles({ [filenames[0]]: true })
        }
      } catch (err) {
        if (!cancelled) setError(err.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [collectionName])

  const toggleFile = useCallback((filename) => {
    setExpandedFiles((prev) => ({ ...prev, [filename]: !prev[filename] }))
  }, [])

  const toggleChunk = useCallback((chunkId) => {
    setExpandedChunks((prev) => ({ ...prev, [chunkId]: !prev[chunkId] }))
  }, [])

  // 删除文档
  const handleDelete = useCallback(async (docId, filename) => {
    if (!confirm(`确定要删除 "${filename}" 吗？\n该操作会同时删除 OSS 上的原始文件，不可恢复。`)) {
      return
    }
    setDeleting(docId)
    try {
      await deleteDocument(docId)
      // 刷新列表
      const docsRes = await getCollectionDocuments(collectionName)
      setDocuments(docsRes.data || [])
    } catch (err) {
      alert('删除失败: ' + (err.message || err))
    } finally {
      setDeleting(null)
    }
  }, [collectionName])

  const totalChunks = chunks.length
  const fileCount = Object.keys(grouped).length

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content collection-modal" onClick={(e) => e.stopPropagation()}>
        {/* 头部 */}
        <div className="modal-header">
          <div className="modal-header-left">
            <span className="modal-icon">📦</span>
            <div>
              <div className="modal-title">{collectionName}</div>
              <div className="modal-subtitle">{documents.length} 个文件 · {totalChunks} 个文档块</div>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Tab 切换 */}
        <div className="modal-tabs">
          <button
            className={`modal-tab ${tab === 'chunks' ? 'active' : ''}`}
            onClick={() => setTab('chunks')}
          >
            📄 文档块 ({totalChunks})
          </button>
          <button
            className={`modal-tab ${tab === 'documents' ? 'active' : ''}`}
            onClick={() => setTab('documents')}
          >
            🗃️ 文件管理 ({documents.length})
          </button>
        </div>

        {/* 加载中 */}
        {loading && (
          <div className="modal-loading">
            <div className="typing-indicator"><span /><span /><span /></div>
            <p>正在加载...</p>
          </div>
        )}

        {/* 错误 */}
        {error && <div className="modal-error">⚠️ {error}</div>}

        {/* 空状态 */}
        {!loading && !error && chunks.length === 0 && documents.length === 0 && (
          <div className="modal-empty"><p>该集合中没有数据</p></div>
        )}

        {/* ======== 文档块 Tab ======== */}
        {!loading && !error && tab === 'chunks' && chunks.length > 0 && (
          <div className="modal-body">
            {Object.entries(grouped).map(([filename, fileChunks]) => (
              <div key={filename} className="chunk-group">
                <div className="chunk-group-header" onClick={() => toggleFile(filename)}>
                  <span className="chunk-group-icon">{expandedFiles[filename] ? '▼' : '▶'}</span>
                  <span className="chunk-group-name">📄 {filename}</span>
                  <span className="chunk-group-count">{fileChunks.length} 块</span>
                </div>
                {expandedFiles[filename] && (
                  <div className="chunk-group-body">
                    {fileChunks.map((chunk) => {
                      const isExpanded = expandedChunks[chunk.id]
                      const preview = chunk.content ? chunk.content.slice(0, 120) : ''
                      const meta = chunk.metadata || {}
                      return (
                        <div key={chunk.id} className="chunk-item">
                          <div className="chunk-meta">
                            {meta.chunk_index !== undefined && (
                              <span className="chunk-index">第 {meta.chunk_index + 1}/{meta.chunk_total} 块</span>
                            )}
                            {meta.page && <span className="chunk-page">p.{meta.page}</span>}
                            <span className="chunk-size">{chunk.content?.length || 0} 字</span>
                          </div>
                          <div className="chunk-content">
                            {isExpanded || chunk.content?.length <= 120 ? chunk.content : preview + '...'}
                          </div>
                          {chunk.content?.length > 120 && (
                            <button className="chunk-toggle" onClick={() => toggleChunk(chunk.id)}>
                              {isExpanded ? '收起 ▲' : '展开全文 ▼'}
                            </button>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* ======== 文件管理 Tab ======== */}
        {!loading && !error && tab === 'documents' && (
          <div className="modal-body">
            {documents.length === 0 ? (
              <div className="modal-empty"><p>暂无文件记录</p></div>
            ) : (
              <div className="doc-list">
                {documents.map((doc) => (
                  <div key={doc.id} className="doc-item">
                    <div className="doc-item-info">
                      <div className="doc-item-name">📄 {doc.filename}</div>
                      <div className="doc-item-meta">
                        <span>类型: {doc.file_type || '未知'}</span>
                        <span>大小: {(doc.file_size / 1024).toFixed(1)}KB</span>
                        <span>分块: {doc.chunk_count} 个</span>
                        <span>上传: {doc.created_at ? doc.created_at.slice(0, 10) : '未知'}</span>
                      </div>
                    </div>
                    <button
                      className="btn btn-sm btn-danger"
                      onClick={() => handleDelete(doc.id, doc.filename)}
                      disabled={deleting === doc.id}
                    >
                      {deleting === doc.id ? '删除中...' : '🗑️ 删除'}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* 底部 */}
        <div className="modal-footer">
          {tab === 'chunks' ? (
            <span className="modal-footer-info">共 {totalChunks} 个文档块</span>
          ) : (
            <span className="modal-footer-info">共 {documents.length} 个文件</span>
          )}
          <button className="btn btn-sm" onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  )
}