import React, { useState } from 'react'
import { changePassword } from '../services/api'

export default function ChangePasswordModal({ onClose }) {
  const [oldPwd, setOldPwd] = useState('')
  const [newPwd, setNewPwd] = useState('')
  const [confirmPwd, setConfirmPwd] = useState('')
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)

    if (newPwd.length < 6) {
      setError('新密码至少 6 位')
      return
    }
    if (newPwd !== confirmPwd) {
      setError('两次输入的新密码不一致')
      return
    }

    setSubmitting(true)
    try {
      await changePassword(oldPwd, newPwd)
      setSuccess(true)
      setOldPwd('')
      setNewPwd('')
      setConfirmPwd('')
    } catch (err) {
      setError(err.message || '修改密码失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content user-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-left">
            <span className="modal-icon">🔑</span>
            <div>
              <div className="modal-title">修改密码</div>
              <div className="modal-subtitle">修改当前账号的登录密码</div>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          {success && (
            <div className="modal-success" style={{ padding: 12, background: 'var(--color-success)', color: '#fff', borderRadius: 6, marginBottom: 12, fontSize: 13 }}>
              ✅ 密码修改成功！
            </div>
          )}

          <form className="user-form" onSubmit={handleSubmit}>
            <div className="form-field">
              <label>原密码 *</label>
              <input
                type="password"
                value={oldPwd}
                onChange={(e) => setOldPwd(e.target.value)}
                required
              />
            </div>
            <div className="form-field">
              <label>新密码 *（至少 6 位）</label>
              <input
                type="password"
                value={newPwd}
                onChange={(e) => setNewPwd(e.target.value)}
                required
              />
            </div>
            <div className="form-field">
              <label>确认新密码 *</label>
              <input
                type="password"
                value={confirmPwd}
                onChange={(e) => setConfirmPwd(e.target.value)}
                required
              />
            </div>
            {error && <div className="form-error">⚠️ {error}</div>}
            <div className="form-actions">
              <button type="button" className="btn btn-sm" onClick={onClose}>关闭</button>
              <button type="submit" className="btn btn-sm btn-primary" disabled={submitting}>
                {submitting ? '提交中...' : '确认修改'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
