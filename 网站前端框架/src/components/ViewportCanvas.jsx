// 布局由 CSS 根据视口和主内容可用宽度调整，图表保留各自的坐标与比例。
export default function ViewportCanvas({ children }) {
  return (
    <div className="app-viewport">
      <div className="app-canvas">{children}</div>
    </div>
  );
}
