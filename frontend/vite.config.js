import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        // 显式用 IPv4，避免 Node 18+ 将 localhost 解析为 ::1 (IPv6)
        // 而后端 uvicorn 只监听 0.0.0.0 (IPv4) 时导致的 ECONNREFUSED
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
