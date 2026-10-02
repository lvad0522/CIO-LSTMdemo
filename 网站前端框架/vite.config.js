import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import process from 'node:process'

// https://vite.dev/config/
export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiBase = env.VITE_API_BASE?.trim()

  // 双站点构建不能靠 App.jsx 的相对地址回退：产物一旦放错站点会静默连到
  // 错误后端。开发服务器仍允许走本地 proxy；正式 build 必须显式钉死 API。
  if (command === 'build' && !apiBase) {
    throw new Error(
      'vite build 必须显式设置 VITE_API_BASE，例如 '
      + 'VITE_API_BASE=http://127.0.0.1:8002 npm run build',
    )
  }

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  }
})
