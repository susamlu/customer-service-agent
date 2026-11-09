import { type RefObject, useState } from 'react';
import type { Message } from './types';

interface Props {
  messages: Message[];
  busy: boolean;
  suggestions: { label: string; query: string }[];
  bottomRef: RefObject<HTMLDivElement>;
  onSend: (query: string) => void;
}

const ROUTE_LABEL: Record<string, string> = {
  knowledge: '知识库问答',
  order: '订单查询',
  human: '转人工',
  clarify: '澄清追问',
};

export default function ChatPanel({ messages, busy, suggestions, bottomRef, onSend }: Props) {
  const [input, setInput] = useState('');
  const empty = messages.length === 0;

  const submit = () => {
    const query = input.trim();
    if (!query || busy) return;
    setInput('');
    onSend(query);
  };

  return (
    <section className="chat">
      <div className="messages">
        {empty && (
          <div className="empty-state">
            <h2>你好，我是电商客服助手</h2>
            <p>每次回答都会在右侧展示完整链路：意图识别 → 订单/知识并行 → 上下文拼装 → 流式生成。</p>
            <div className="suggestions">
              {suggestions.map((s) => (
                <button key={s.label} onClick={() => onSend(s.query)} disabled={busy}>
                  <span className="suggestion-tag">{s.label}</span>
                  {s.query}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`bubble-row ${m.role}`}>
            <div className={`bubble ${m.role} ${m.error ? 'error' : ''}`}>
              {m.text}
              {m.streaming && (
                <span className="typing-dots" role="status" aria-label="正在输入">
                  <span />
                  <span />
                  <span />
                </span>
              )}
              {m.order && (
                <div className="order-card">
                  <div className="order-card-head">
                    <span className="order-id">{m.order.orderId}</span>
                    <span className="order-status">{m.order.statusLabel}</span>
                  </div>
                  <div className="order-line">
                    <span className="order-label">商品</span>
                    <span>{m.order.items}</span>
                  </div>
                  <div className="order-line">
                    <span className="order-label">最新物流</span>
                    <span>{m.order.latestTrace}</span>
                  </div>
                  <div className="order-line">
                    <span className="order-label">下单</span>
                    <span>{m.order.placedAt}</span>
                  </div>
                </div>
              )}
              {!m.streaming && m.role === 'assistant' && m.text && (m.route || m.domain) && (
                <div className="bubble-meta">
                  {m.route && <span>{ROUTE_LABEL[m.route] ?? m.route}</span>}
                  {m.domain && <span>主题 · {m.domain}</span>}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="composer">
        <input
          value={input}
          placeholder={busy ? '回答生成中…' : '输入客服问题，例如：退款多久到账'}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
          disabled={busy}
        />
        <button className="btn-send" onClick={submit} disabled={busy || !input.trim()}>
          发送
        </button>
      </div>
    </section>
  );
}
