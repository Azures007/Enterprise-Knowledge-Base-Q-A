import React, { useState, useRef, useCallback, useEffect } from 'react'
import { uploadDocument } from '../services/api'

const MAX_CONCURRENT = 3  // 最大并发上传数

export default function UploadQueue({ uploadCollection, onUploadSuccess, serverOnline }) {
  const [tasks, setTasks] = useState([])  // { id, file, filename, progress, status: 'waiting'|'uploading'|'done'|'error', result, warning }
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)
  const taskIdRef = useRef(0)

  // 获取下一个任务 ID
  const nextId = useCallback(() => ++taskIdRef.current, [])

  // 添加文件到队列
  const addFiles = useCallback((fileList) => {
    const allowed = ['.pdf', '.docx', '.doc', '.docm', '.pptx',
      '.xlsx', '.xls', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
      '.txt', '.md', '.py', '.yaml', '.yml', '.html', '.htm', '.xml', '.csv', '.json',
      '.wps', '.et']
    const maxSize = 100 * 1024 * 1024
    const getExt = (name) => '.' + (name.split('.').pop() || '').toLowerCase()

    const newTasks = []
    for (const file of fileList) {
      const ext = getExt(file.name)

      // 前端校验
      let error = null
      if (!ext) error = { type: 'invalid_extension', message: '文件没有扩展名' }
      else if (!allowed.includes(ext)) error = { type: 'unsupported_format', message: `不支持 "${ext}" 格式` }
      else if (file.size === 0) error = { type: 'empty_file', message: '文件为空' }
      else if (file.size > maxSize) error = { type: 'file_too_large', message: `文件过大 (${(file.size / 1024 / 1024).toFixed(1)}MB)` }

      newTasks.push({
        id: nextId(),
        file,
        filename: file.name,
        progress: 0,
        status: error ? 'error' : 'waiting',
        result: null,
        warning: null,
        error,
      })
    }

    setTasks((prev) => [...prev, ...newTasks])
  }, [nextId])

  // 删除单个任务
  const removeTask = useCallback((taskId) => {
    setTasks((prev) => prev.filter((t) => t.id !== taskId))
  }, [])

  // 清空已完成的任务
  const clearDone = useCallback(() => {
    setTasks((prev) => prev.filter((t) => t.status === 'waiting' || t.status === 'uploading'))
  }, [])

  // 更新任务状态
  const updateTask = useCallback((taskId, updates) => {
    setTasks((prev) => prev.map((t) => (t.id === taskId ? { ...t, ...updates } : t)))
  }, [])

  // 执行上传（处理队列中的 waiting 任务）
  useEffect(() => {
    const uploading = tasks.filter((t) => t.status === 'uploading').length
    const waiting = tasks.filter((t) => t.status === 'waiting')

    if (uploading >= MAX_CONCURRENT || waiting.length === 0) return

    // 启动尽可能多的任务
    const slots = MAX_CONCURRENT - uploading
    for (let i = 0; i < Math.min(slots, waiting.length); i++) {
      const task = waiting[i]
      startUpload(task)
    }
  }, [tasks])

  // 上传单个文件
  const startUpload = async (task) => {
    updateTask(task.id, { status: 'uploading', progress: 0 })

    try {
      const result = await uploadDocument(
        task.file,
        (progress) => updateTask(task.id, { progress }),
        null,
        uploadCollection,
      )

      const warnings = result.data?.warnings
      updateTask(task.id, {
        status: 'done',
        progress: 100,
        result: result.data,
        warning: warnings?.length > 0 ? warnings : null,
      })
      onUploadSuccess?.()
    } catch (err) {
      const msg = err._isStructured ? (err.message || err.suggestion) : (err.message || '上传失败')
      updateTask(task.id, {
        status: 'error',
        error: { type: 'upload_failed', message: msg },
      })
    }
  }

  // 拖拽事件
  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) addFiles(files)
  }, [addFiles])

  const handleDragOver = (e) => { e.preventDefault(); setDragOver(true) }
  const handleDragLeave = () => setDragOver(false)

  // 文件选择
  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files)
    if (files.length > 0) addFiles(files)
    e.target.value = ''
  }

  // 状态图标
  const statusIcon = (task) => {
    switch (task.status) {
      case 'waiting': return '⏳'
      case 'uploading': return '↑'
      case 'done': return '✅'
      case 'error': return '❌'
      default: return '📄'
    }
  }

  // 状态文本
  const statusText = (task) => {
    if (task.status === 'done') {
      const chunks = task.result?.chunks_added ?? 0
      return `导入成功 (${chunks} 块)`
    }
    if (task.status === 'error') {
      return task.error?.message || '上传失败'
    }
    if (task.status === 'waiting') return '等待上传'
    if (task.status === 'uploading') return `上传中 ${task.progress}%`
    return ''
  }

  const hasTasks = tasks.length > 0
  const doneCount = tasks.filter((t) => t.status === 'done' || t.status === 'error').length

  return (
    <div className="upload-queue">
      {/* 上传区域 */}
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="upload-icon">📄</div>
        <div className="upload-text">拖拽文件到此处或点击上传</div>
        <div className="upload-hint">支持 PDF / Word / PPT / TXT / MD / Excel / 图片</div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.doc,.pptx,.xlsx,.xls,.jpg,.jpeg,.png,.bmp,.tiff,.txt,.md,.json,.csv,.html,.wps,.et"
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />
      </div>

      {/* 上传任务列表 */}
      {hasTasks && (
        <div className="upload-task-list">
          <div className="upload-task-header">
            <span className="upload-task-title">上传队列 ({tasks.length})</span>
            {doneCount > 0 && doneCount < tasks.length && (
              <button className="upload-task-clear-btn" onClick={clearDone}>清除已完成</button>
            )}
          </div>

          {tasks.map((task) => (
            <div key={task.id} className={`upload-task ${task.status}`}>
              <div className="upload-task-main">
                <span className="upload-task-icon">{statusIcon(task)}</span>
                <div className="upload-task-info">
                  <div className="upload-task-name">{task.filename}</div>
                  <div className="upload-task-status">{statusText(task)}</div>
                </div>
                {(task.status === 'done' || task.status === 'error') && (
                  <button className="upload-task-remove" onClick={() => removeTask(task.id)}>✕</button>
                )}
              </div>

              {/* 进度条 */}
              {task.status === 'uploading' && (
                <div className="upload-task-progress-bar">
                  <div className="upload-task-progress-fill" style={{ width: `${task.progress}%` }} />
                </div>
              )}

              {/* 警告信息 */}
              {task.warning && task.warning.length > 0 && (
                <div className="upload-task-warning">
                  ⚠️ {task.warning.join('; ')}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}