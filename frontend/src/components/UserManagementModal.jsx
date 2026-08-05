import React, { useState, useEffect, useCallback } from 'react'
import { getUsers, createUser, deleteUser } from '../services/api'

export default function UserManagementModal({ onClose }) {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState({ username: '', password: '', display_name: '', is_admin: false })
  const [formError, setFormError] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [deletingId, setDeletingId] = useState(null)

  const loadUsers = useCallback(async () => {
    try {
      setLoading(true)
      const res = await getUsers()
      setUsers(res.data || [])
      setError(null)
    } catch (err) {
      setError(err.message || '加载用户失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadUsers()
  }, [loadUsers])

  const handleCreate = async (e) => {
    e.preventDefault()
    setFormError(null)
    setSubmitting(true)
    try {
      await createUser({
        username: form.username.trim(),
        password: form.password,
        display_name: form.display_name?.trim() || null,
        is_admin: form.is_admin,
      })
      setShowForm(false)
      setForm({ username: '', password: '', display_name: '', is_admin: false })
      await loadUsers()
    } catch (err) {
      setFormError(err.message || '创建用户失败')
    } finally {
      setSubmitting(false)
    }
  }

  const handleDelete = async (userId, username) => {
    if (!confirm(`确定要删除用户 "${username}" 吗？\n该用户的所有对话也会被删除。`)) return
    setDeletingId(userId)
    try {
      await deleteUser(userId)
      await loadUsers()
    } catch (err) {
      alert('删除失败: ' + (err.message || err))
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content user-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-left">
            <span className="modal-icon">👥</span>
            <div>
              <div className="modal-title">用户管理</div>
              <div className="modal-subtitle">{users.length} 个用户</div>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          {error && <div className="modal-error">⚠️ {error}</div>}

          {/* 创建用户按钮 */}
          {!showForm && (
            <button className="btn btn-sm btn-primary" onClick={() => setShowForm(true)} style={{ marginBottom: 12 }}>
              ➕ 创建用户
            </button>
          )}

          {/* 创建表单 */}
          {showForm && (
            <form className="user-form" onSubmit={handleCreate}>
              <div className="form-field">
                <label>用户名 *</label>
                <input
                  value={form.username}
                  onChange={(e) => setForm({ ...form, username: e.target.value })}
                  placeholder="如：张三"
                  required
                />
              </div>
              <div className="form-field">
                <label>密码 *（至少 6 位）</label>
                <input
                  type="password"
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  placeholder="设置初始密码"
                  required
                />
              </div>
              <div className="form-field">
                <label>显示名</label>
                <input
                  value={form.display_name}
                  onChange={(e) => setForm({ ...form, display_name: e.target.value })}
                  placeholder="可选"
                />
              </div>
              <div className="form-field form-checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={form.is_admin}
                    onChange={(e) => setForm({ ...form, is_admin: e.target.checked })}
                  />
                  设为管理员
                </label>
              </div>
              {formError && <div className="form-error">⚠️ {formError}</div>}
              <div className="form-actions">
                <button type="button" className="btn btn-sm" onClick={() => { setShowForm(false); setFormError(null) }}>
                  取消
                </button>
                <button type="submit" className="btn btn-sm btn-primary" disabled={submitting}>
                  {submitting ? '创建中...' : '创建'}
                </button>
              </div>
            </form>
          )}

          {/* 用户列表 */}
          {loading ? (
            <div className="modal-loading"><p>加载中...</p></div>
          ) : (
            <div className="user-list">
              {users.map((u) => (
                <div key={u.id} className="user-item">
                  <div className="user-item-info">
                    <div className="user-item-name">
                      {u.display_name || u.username}
                      {u.is_admin && <span className="user-badge">管理员</span>}
                    </div>
                    <div className="user-item-meta">
                      <span>@{u.username}</span>
                      {u.created_at && <span>创建于 {u.created_at.slice(0, 10)}</span>}
                    </div>
                  </div>
                  {!u.is_admin && (
                    <button
                      className="btn btn-sm btn-danger"
                      onClick={() => handleDelete(u.id, u.username)}
                      disabled={deletingId === u.id}
                    >
                      {deletingId === u.id ? '删除中...' : '🗑️ 删除'}
                    </button>
                  )}
                </div>
              ))}
              {users.length === 0 && <div className="modal-empty"><p>暂无用户</p></div>}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn btn-sm" onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  )
}
