/**
 * API 服务层 - 封装与后端的所有通信
 */

const BASE_URL = '/api'

/**
 * 通用请求封装
 */
async function request(url, options = {}) {
  const config = {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  }

  const response = await fetch(`${BASE_URL}${url}`, config)

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    // 处理限流错误（429）
    if (response.status === 429 || errorData.code === -1) {
      const error = new Error(errorData.message || '请求过于频繁，请稍后再试')
      error._isStructured = true
      error.error_type = errorData.data?.error_type || 'rate_limited'
      error.retry_after = errorData.data?.retry_after || 1
      throw error
    }
    throw new Error(errorData.detail || `请求失败 (${response.status})`)
  }

  return response.json()
}

/**
 * 健康检查
 */
export async function checkHealth() {
  return request('/health')
}

/**
 * 知识库问答（非流式）
 */
export async function queryKB({ question, k = 5, concise = false }) {
  return request('/query', {
    method: 'POST',
    body: JSON.stringify({ question, k, concise }),
  })
}

/**
 * 知识库问答（流式 SSE）
 * 返回 EventSource 实例，外部通过 onmessage 处理回调
 */
export function streamQuery({ question, k = 5, concise = false, collection = null }, callbacks, signal) {
  const { onChunk, onSources, onMeta, onDone, onError, onAbort } = callbacks

  const body = { question, k, concise }
  if (collection) body.collection = collection

  fetch(`${BASE_URL}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        if (response.status === 429) {
          const err = await response.json().catch(() => ({}))
          onError?.(err.message || '请求过于频繁，请稍后再试')
          return
        }
        const err = await response.json().catch(() => ({}))
        onError?.(err.detail || `请求失败 (${response.status})`)
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const dataStr = line.slice(6).trim()
          if (!dataStr) continue

          try {
            const data = JSON.parse(dataStr)
            switch (data.type) {
              case 'chunk':
                onChunk?.(data.data)
                break
              case 'meta':
                // 新格式：合并 sources + answer_type
                onMeta?.(data)
                break
              case 'sources':
                // 兼容旧格式
                onSources?.(data.data)
                break
              case 'done':
                onDone?.(data.data)
                break
              case 'error':
                onError?.(data.data)
                break
            }
          } catch {
            // 忽略解析失败的行
          }
        }
      }
    })
    .catch((err) => {
      if (err.name === 'AbortError') {
        onAbort?.()
      } else {
        onError?.(err.message || '网络连接失败')
      }
    })
}

// 检查响应是否为 429 限流
function isRateLimited(response) {
  return response.status === 429
}

/**
 * 获取 OSS 直传签名 URL
 */
export async function getUploadToken(filename) {
  return request(`/upload/token?filename=${encodeURIComponent(filename)}`)
}

/**
 * 确认 OSS 直传完成并导入知识库
 */
export async function confirmUpload({ object_key, filename, collection }) {
  return request('/ingest/confirm', {
    method: 'POST',
    body: JSON.stringify({ object_key, filename, collection }),
  })
}

/**
 * 上传文档到知识库（支持 OSS 直传和传统上传两种方式）
 */
export async function uploadDocument(file, onProgress, targetFilename, targetCollection) {
  const finalName = targetFilename || file.name
  const params = []

  // 尝试 OSS 直传（获取签名 URL → 直接上传到 OSS → 确认）
  try {
    const tokenRes = await getUploadToken(finalName)
    const { upload_url, object_key } = tokenRes.data

    // 直接上传到 OSS（XHR 可监听到真实进度）
    const uploadResult = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(true)
        } else {
          reject({ error_type: 'oss_upload_failed', message: `OSS 上传失败 (${xhr.status})`, suggestion: '请重试', _isStructured: true })
        }
      }

      xhr.onerror = () => reject({ error_type: 'network_error', message: 'OSS 上传网络错误', suggestion: '请检查网络', _isStructured: true })

      xhr.open('PUT', upload_url)
      xhr.setRequestHeader('Content-Type', 'application/octet-stream')
      xhr.send(file)
    })

    // 上传完成，确认并导入
    if (onProgress) onProgress(100)
    const confirmResult = await confirmUpload({
      object_key,
      filename: finalName,
      collection: targetCollection,
    })
    return confirmResult
  } catch (err) {
    // OSS 直传失败（如 CORS 问题、网络问题），降级到传统上传
    console.warn('OSS 直传失败，降级到传统上传:', err.message || err)
  }

  // 传统方式：上传到后端
  const formData = new FormData()
  formData.append('file', file, finalName)
  if (targetFilename) formData.append('filename', targetFilename)
  if (targetCollection) formData.append('collection', targetCollection)

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    }

    xhr.onload = () => {
      try {
        const data = JSON.parse(xhr.responseText)
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(data)
        } else {
          const detail = data.detail
          if (typeof detail === 'object' && detail !== null) {
            reject({ ...detail, _isStructured: true })
          } else {
            reject({
              error_type: 'unknown',
              message: detail || `上传失败 (${xhr.status})`,
              suggestion: '请重试或联系管理员',
              _isStructured: true,
            })
          }
        }
      } catch {
        reject({
          error_type: 'parse_error',
          message: '服务器响应解析失败',
          suggestion: '请检查后端服务是否正常运行',
          _isStructured: true,
        })
      }
    }

    xhr.onerror = () => reject({
      error_type: 'network_error',
      message: '网络连接失败',
      suggestion: '请检查网络连接和后端服务状态',
      _isStructured: true,
    })
    xhr.upload.onerror = () => reject({
      error_type: 'network_error',
      message: '上传过程中网络中断',
      suggestion: '请检查网络连接后重试',
      _isStructured: true,
    })

    xhr.open('POST', `${BASE_URL}/ingest`)
    xhr.send(formData)
  })
}

/**
 * 查看知识库统计
 */
export async function getStats() {
  return request('/stats')
}

/**
 * 查看集合列表
 */
export async function getCollections() {
  return request('/collections')
}

/**
 * 查看集合中的文档块列表
 */
export async function getCollectionChunks(name = 'knowledge_base', limit = 500, offset = 0) {
  return request(`/collections/${encodeURIComponent(name)}/chunks?limit=${limit}&offset=${offset}`)
}

/**
 * 获取集合中的文档列表
 */
export async function getCollectionDocuments(name = 'knowledge_base') {
  return request(`/collections/${encodeURIComponent(name)}/documents`)
}

/**
 * 创建新集合
 */
export async function createCollection(name) {
  return request('/collections', {
    method: 'POST',
    body: JSON.stringify({ name }),
  })
}

/**
 * 重命名集合
 */
export async function renameCollection(oldName, newName) {
  return request(`/collections/${encodeURIComponent(oldName)}`, {
    method: 'PUT',
    body: JSON.stringify({ name: newName }),
  })
}

/**
 * 删除文档（含 OSS 原文件）
 */
export async function deleteDocument(docId) {
  return request(`/documents/${docId}`, { method: 'DELETE' })
}

// ============================================================
// 对话管理 API
// ============================================================

/**
 * 获取所有对话列表
 */
export async function getConversations() {
  return request('/conversations')
}

/**
 * 创建新对话
 */
export async function createConversation() {
  return request('/conversations', { method: 'POST' })
}

/**
 * 删除对话
 */
export async function deleteConversation(convId) {
  return request(`/conversations/${convId}`, { method: 'DELETE' })
}

/**
 * 批量删除对话
 */
export async function batchDeleteConversations(ids) {
  return request('/conversations', {
    method: 'DELETE',
    body: JSON.stringify({ ids }),
  })
}

/**
 * 修改对话标题
 */
export async function updateConversationTitle(convId, title) {
  return request(`/conversations/${convId}/title`, {
    method: 'PUT',
    body: JSON.stringify({ title }),
  })
}

/**
 * 获取对话中的消息列表
 */
export async function getConversationMessages(convId) {
  return request(`/conversations/${convId}/messages`)
}

/**
 * 向对话中添加消息
 */
export async function addConversationMessage(convId, { role, content, sources, answer_type }) {
  return request(`/conversations/${convId}/messages`, {
    method: 'POST',
    body: JSON.stringify({ role, content, sources, answer_type }),
  })
}

/**
 * 更新消息内容（流式完成后）
 */
export async function updateMessageContent(convId, msgId, { content, sources }) {
  return request(`/conversations/${convId}/messages/${msgId}`, {
    method: 'PUT',
    body: JSON.stringify({ content, sources }),
  })
}

/**
 * 删除集合
 */
export async function deleteCollection(name = 'knowledge_base') {
  return request(`/collections/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
}
