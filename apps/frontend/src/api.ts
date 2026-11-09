import type { Message, StreamEvent } from './types';

export function getSessionId(): string {
  const KEY = 'csagent-session';
  let id = localStorage.getItem(KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(KEY, id);
  }
  return id;
}

export async function fetchHistory(sessionId: string): Promise<Message[]> {
  const resp = await fetch(`/api/history?sessionId=${encodeURIComponent(sessionId)}`);
  if (!resp.ok) return [];
  return resp.json();
}

export async function clearHistory(sessionId: string): Promise<void> {
  await fetch(`/api/history?sessionId=${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
}

export async function fetchModelStatus(): Promise<{ status: string }> {
  try {
    const resp = await fetch('/api/model-status');
    if (!resp.ok) return { status: 'down' };
    return resp.json();
  } catch {
    return { status: 'down' };
  }
}

export interface UserSummary {
  userId: string;
  name: string;
  tier: string;
}

/** mock 用户列表，供顶栏切换。 */
export async function fetchUsers(): Promise<UserSummary[]> {
  try {
    const resp = await fetch('/api/mock/users');
    if (!resp.ok) return [];
    return resp.json();
  } catch {
    return [];
  }
}

/** 当前用户的订单号，用于把默认问题里的订单号换成该用户自己的。 */
export async function fetchUserOrders(userId: string): Promise<string[]> {
  try {
    const resp = await fetch(`/api/mock/context?userId=${encodeURIComponent(userId)}`);
    if (!resp.ok) return [];
    const ctx = (await resp.json()) as { orders?: { orderId: string }[] };
    return (ctx.orders ?? []).map((order) => order.orderId);
  } catch {
    return [];
  }
}

/** POST + SSE：逐事件回调。 */
export async function streamChat(
  sessionId: string,
  query: string,
  userId: string,
  onEvent: (event: StreamEvent) => void
): Promise<void> {
  const resp = await fetch(`/api/chat?sessionId=${encodeURIComponent(sessionId)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, userId }),
  });
  if (!resp.body) {
    onEvent({ type: 'error', message: `网关返回 ${resp.status}` });
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const dataLine = block.split('\n').find((line) => line.startsWith('data: '));
      if (!dataLine) continue;
      try {
        onEvent(JSON.parse(dataLine.slice(6)) as StreamEvent);
      } catch {
        // 忽略无法解析的块
      }
    }
  }
}
