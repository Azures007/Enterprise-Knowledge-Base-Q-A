import React, { useState, useEffect, useCallback, useRef } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import LoginPage from './components/LoginPage'
import UserManagementModal from './components/UserManagementModal'
import ChangePasswordModal from './components/ChangePasswordModal'
import AuditModal from './components/AuditModal'
import TraceModal from './components/TraceModal'
import {
  checkHealth,
  getCollections,
  getConversations,
  createConversation,
  getConversationMessages,
  isLoggedIn,
  isAdmin,
  clearToken,
  setUnauthorizedHandler,
} from './services/api'

export default function App() {
  // 认证
  const [authed, setAuthed] = useState(isLoggedIn())
  const [showUserModal, setShowUserModal] = useState(false)
  const [showPwdModal, setShowPwdModal] = useState(false)
  const [showAuditModal, setShowAuditModal] = useState(false)
  const [showTraceModal, setShowTraceModal] = useState(false)

  // 知识库
  const [collections, setCollections] = useState([])
  const [selectedCollection, setSelectedCollection] = useState(null)
  const [serverOnline, setServerOnline] = useState(false)
  const [error, setError] = useState(null)

  // 对话
  const [conversations, setConversations] = useState([])
  const [currentConvId, setCurrentConvId] = useState(null)
  const [messages, setMessages] = useState([])

  const initRef = useRef(false)
  const creatingRef = useRef(false)

  // 注册 401 处理：token 失效时回到登录页
  useEffect(() => {
    setUnauthorizedHandler(() => {
      setAuthed(false)
    })
  }, [])

  // ================================================================
  // 初始化（登录成功后执行）
  // ================================================================
  useEffect(() => {
    if (!authed || initRef.current) return
    initRef.current = true

    async function init() {
      try {
        await checkHealth()
        setServerOnline(true)

        // 加载知识库信息
        await refreshKBInfo()

        // 加载对话列表
        await loadConversations()
      } catch (err) {
        // 401 时 setUnauthorizedHandler 会登出；其余错误标记离线
        if (!err._unauthorized) {
          setServerOnline(false)
          setError(`无法连接到后端服务: ${err.message}`)
        }
      }
    }
    init()
  }, [authed])

  // 登出
  const handleLogout = useCallback(() => {
    clearToken()
    setAuthed(false)
    // 清空主界面状态
    setCollections([])
    setConversations([])
    setCurrentConvId(null)
    setMessages([])
    initRef.current = false
  }, [])

  // 登录成功
  const handleLogin = useCallback(() => {
    setAuthed(true)
  }, [])

  // ================================================================
  // 知识库
  // ================================================================
  const refreshKBInfo = useCallback(async () => {
    try {
      const collectionsRes = await getCollections()
      setCollections(collectionsRes.data.collections || [])
    } catch (err) {
      console.warn('获取知识库信息失败:', err.message)
    }
  }, [])

  // ================================================================
  // 对话管理
  // ================================================================

  // 加载对话列表
  const loadConversations = useCallback(async () => {
    try {
      const res = await getConversations()
      const list = res.data || []
      setConversations(list)

      // 如果有对话，选中第一个
      if (list.length > 0) {
        switchConversation(list[0].id)
      } else {
        // 没有对话则创建一个
        handleCreateConversation()
      }
    } catch (err) {
      console.warn('加载对话列表失败:', err.message)
    }
  }, [])

  // 切换对话
  const switchConversation = useCallback(async (convId) => {
    if (convId === null) {
      setCurrentConvId(null)
      setMessages([])
      return
    }

    setCurrentConvId(convId)
    try {
      const res = await getConversationMessages(convId)
      setMessages(res.data || [])
    } catch (err) {
      console.warn('加载对话消息失败:', err.message)
      setMessages([])
    }
  }, [])

  // 创建新对话
  const handleCreateConversation = useCallback(async () => {
    if (creatingRef.current) return
    creatingRef.current = true
    try {
      const res = await createConversation()
      const newConv = res.data
      setConversations((prev) => [newConv, ...prev])
      setCurrentConvId(newConv.id)
      setMessages([])
    } catch (err) {
      console.error('创建对话失败:', err)
    } finally {
      creatingRef.current = false
    }
  }, [])

  // 刷新对话列表（发消息后更新消息数）
  const refreshConversations = useCallback(async () => {
    try {
      const res = await getConversations()
      setConversations(res.data || [])
    } catch (err) {
      console.warn('刷新对话列表失败:', err.message)
    }
  }, [])

  // 对话列表变化回调（由 ConversationPanel 触发）
  const handleConversationsChange = useCallback((updater) => {
    setConversations(updater)
  }, [])

  // 添加消息（本地 + 持久化）
  const addMessage = useCallback(
    (msg) => {
      setMessages((prev) => [...prev, msg])
    },
    [],
  )

  // 更新最后一条消息（本地）
  const updateLastMessage = useCallback((updater) => {
    setMessages((prev) => {
      const copy = [...prev]
      if (copy.length > 0) {
        copy[copy.length - 1] = updater(copy[copy.length - 1])
      }
      return copy
    })
  }, [])

  // 清空对话（仅本地）
  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  // 上传成功后刷新知识库信息
  const handleUploadSuccess = useCallback(() => {
    refreshKBInfo()
  }, [refreshKBInfo])

  // 未登录：显示登录页
  if (!authed) {
    return <LoginPage onLogin={handleLogin} />
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <span className="header-icon">🏢</span>
          <h1>企业知识库问答系统</h1>
        </div>
        <div className="header-right">
          <span className={`status-dot ${serverOnline ? 'online' : 'offline'}`} />
          <span className="status-text">
            {serverOnline ? '服务正常' : '服务离线'}
          </span>
          {isAdmin() && (
            <button className="logout-button" onClick={() => setShowUserModal(true)} title="用户管理">
              👥 用户
            </button>
          )}
          {isAdmin() && (
            <button className="logout-button" onClick={() => setShowAuditModal(true)} title="查询审计">
              📊 审计
            </button>
          )}
          {isAdmin() && (
            <button className="logout-button" onClick={() => setShowTraceModal(true)} title="链路追踪">
              🔍 追踪
            </button>
          )}
          <button className="logout-button" onClick={() => setShowPwdModal(true)} title="修改密码">
            🔑 改密码
          </button>
          <button className="logout-button" onClick={handleLogout} title="退出登录">
            退出
          </button>
        </div>
      </header>

      {showUserModal && (
        <UserManagementModal onClose={() => setShowUserModal(false)} />
      )}

      {showPwdModal && (
        <ChangePasswordModal onClose={() => setShowPwdModal(false)} />
      )}

      {showAuditModal && (
        <AuditModal onClose={() => setShowAuditModal(false)} />
      )}

      {showTraceModal && (
        <TraceModal onClose={() => setShowTraceModal(false)} />
      )}

      <div className="app-body">
        <Sidebar
          collections={collections}
          conversations={conversations}
          currentConvId={currentConvId}
          serverOnline={serverOnline}
          isAdmin={isAdmin()}
          onUploadSuccess={handleUploadSuccess}
          onClearMessages={clearMessages}
          onRefresh={refreshKBInfo}
          onSwitchConversation={switchConversation}
          onCreateConversation={handleCreateConversation}
          onConversationsChange={handleConversationsChange}
        />

        <ChatArea
          messages={messages}
          addMessage={addMessage}
          updateLastMessage={updateLastMessage}
          currentConvId={currentConvId}
          serverOnline={serverOnline}
          error={error}
          onConversationUpdated={refreshConversations}
          collections={collections}
          selectedCollection={selectedCollection}
          onSelectCollection={setSelectedCollection}
        />
      </div>
    </div>
  )
}
