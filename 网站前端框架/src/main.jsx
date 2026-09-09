import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// 从浏览器 bfcache（前进/后退缓存）恢复时，Vite HMR WebSocket 已被浏览器断开
// 且 Vite 8 客户端不会自动重连，强制刷新以保证热更新继续生效
window.addEventListener('pageshow', (e) => {
  if (e.persisted) {
    window.location.reload();
  }
});

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
