import { useEffect, useMemo, useRef, useState } from 'react';
import {
  clearHistory,
  fetchHistory,
  fetchModelStatus,
  fetchUserOrders,
  fetchUsers,
  getSessionId,
  streamChat,
  type UserSummary,
} from './api';
import ChatPanel from './ChatPanel';
import TracePanel from './TracePanel';
import {
  emptyTrace,
  type Message,
  type OrderResult,
  type StreamEvent,
  type Trace,
} from './types';

type Suggestion = { label: string; query: string };

/** 纯知识类问题，不依赖订单号，任何用户都能问。 */
const KNOWLEDGE_POOL: Suggestion[] = [
  { label: '退款时效', query: '退款多久到账？' },
  { label: '退货流程', query: '商品怎么申请退货？' },
  { label: '售后运费', query: '退货运费谁承担？' },
  { label: '开发票', query: '可以开发票吗？' },
  { label: '价保', query: '商品降价了能保价吗？' },
  { label: '配送周期', query: '一般多久能送到？' },
  { label: '保修', query: '保修期内坏了怎么返修？' },
  { label: '会员权益', query: 'PLUS 会员有哪些权益？' },
];

/** 订单类问题必须带当前用户自己的订单号，否则后端按 ctx.orders 查不到。 */
const ORDER_POOL: ((orderId: string) => Suggestion)[] = [
  (orderId) => ({ label: '查物流', query: `我的订单 ${orderId} 现在到哪了？` }),
  (orderId) => ({ label: '查状态', query: `订单 ${orderId} 现在是什么状态？` }),
  (orderId) => ({
    label: '签收异常',
    query: `我的订单 ${orderId} 显示签收了但我没收到，怎么办？`,
  }),
];

const PARALLEL_POOL: ((orderId: string) => Suggestion)[] = [
  (orderId) => ({
    label: '订单 + 政策',
    query: `我的订单 ${orderId} 现在到哪了，另外退款政策怎么说的？`,
  }),
  (orderId) => ({
    label: '订单 + 时效',
    query: `订单 ${orderId} 什么时候能出库，配送一般多久？`,
  }),
  (orderId) => ({
    label: '订单 + 售后',
    query: `我的订单 ${orderId} 有质量问题，退换货怎么处理？`,
  }),
];

/** 文案必须命中后端 HUMAN_PATTERN，否则不会走转人工。 */
const HUMAN_POOL: Suggestion[] = [
  { label: '转人工', query: '转人工客服' },
  { label: '投诉', query: '我要投诉，你们这服务太差了' },
  { label: '找负责人', query: '帮我找你们的负责人' },
];

/** 32 位确定性伪随机，保证同一 seed 下抽取结果稳定。 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 从池中随机取不重复的 count 项；count 超过池大小时取全部。 */
function pick<T>(rand: () => number, pool: T[], count: number): T[] {
  const rest = [...pool];
  const out: T[] = [];
  while (out.length < count && rest.length) {
    out.push(rest.splice(Math.floor(rand() * rest.length), 1)[0]);
  }
  return out;
}

/**
 * 生成空状态建议。
 *
 * 有订单号时抽「订单 / 订单+并行 / 知识 / 人工」各一张，覆盖四种能力组合；
 * 没订单时退回「三张知识 + 一张人工」，避免出现点了必然查不到的单号。
 * 最后整体打散，让每次刷新、切换用户、新建会话的排列都不同。
 */
function buildSuggestions(orderIds: string[], seed: number): Suggestion[] {
  const rand = mulberry32(seed);
  const orderId = orderIds[0];
  const picked = orderId
    ? [
        ...pick(rand, ORDER_POOL.map((make) => make(orderId)), 1),
        ...pick(rand, PARALLEL_POOL.map((make) => make(orderId)), 1),
        ...pick(rand, KNOWLEDGE_POOL, 1),
        ...pick(rand, HUMAN_POOL, 1),
      ]
    : [...pick(rand, KNOWLEDGE_POOL, 3), ...pick(rand, HUMAN_POOL, 1)];
  return pick(rand, picked, picked.length);
}

const USER_KEY = 'csagent-user';

export default function App() {
  const sessionId = getSessionId();
  const [messages, setMessages] = useState<Message[]>([]);
  const [trace, setTrace] = useState<Trace>(emptyTrace);
  const [busy, setBusy] = useState(false);
  const [modelStatus, setModelStatus] = useState('checking');
  const [users, setUsers] = useState<UserSummary[]>([]);
  const [userId, setUserId] = useState(() => localStorage.getItem(USER_KEY) ?? 'u_chen');
  const [orderIds, setOrderIds] = useState<string[]>([]);
  const [suggestionSeed, setSuggestionSeed] = useState(() => Date.now());
  const bottomRef = useRef<HTMLDivElement>(null);

  const suggestions = useMemo(
    () => buildSuggestions(orderIds, suggestionSeed),
    [orderIds, suggestionSeed]
  );

  // 订单号随用户走：切用户先清空，避免上一用户的单号残留在建议里。
  useEffect(() => {
    let ignore = false;
    fetchUserOrders(userId).then((ids) => {
      if (!ignore) setOrderIds(ids);
    });
    return () => {
      ignore = true;
    };
  }, [userId]);

  useEffect(() => {
    fetchHistory(sessionId).then(setMessages);
    fetchUsers().then((list) => {
      setUsers(list);
      if (list.length && !list.some((u) => u.userId === userId)) {
        setUserId(list[0].userId);
      }
    });
    const timer = setInterval(() => {
      fetchModelStatus().then((s) => setModelStatus(s.status));
    }, 5000);
    fetchModelStatus().then((s) => setModelStatus(s.status));
    return () => clearInterval(timer);
    // userId 仅用于初始化默认值，不需要重新拉取
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async (query: string) => {
    if (busy || !query.trim()) return;
    setBusy(true);
    setMessages((prev) => [
      ...prev,
      { role: 'user', text: query },
      { role: 'assistant', text: '', streaming: true },
    ]);
    setTrace({ ...emptyTrace(), active: true });

    let idx = -1;
    setMessages((prev) => {
      idx = prev.length - 1;
      return prev;
    });

    const updateLast = (patch: Partial<Message>) => {
      setMessages((prev) => {
        if (idx < 0) idx = prev.length - 1;
        const copy = [...prev];
        copy[idx] = { ...copy[idx], ...patch };
        return copy;
      });
    };
    const patchTrace = (patch: Partial<Trace>) => {
      setTrace((prev) => ({ ...prev, ...patch }));
    };
    const setStage = (
      name: keyof Trace['stages'],
      status: 'running' | 'done',
      elapsed?: number
    ) => {
      setTrace((prev) => ({
        ...prev,
        stages: { ...prev.stages, [name]: { status, elapsed } },
      }));
    };
    // 缓冲到 done 再一次性渲染：逐字输出是最明显的 AI 特征，
    // 且订单结果比首个 token 早到，会先蹦出一张悬空卡片。
    let buffer = '';
    let pendingOrder: OrderResult | undefined;
    let finished = false;
    const finish = (text: string, error = false) => {
      finished = true;
      updateLast({ text: text || buffer, order: pendingOrder, streaming: false, error });
    };

    const onEvent = (event: StreamEvent) => {
      switch (event.type) {
        case 'stage':
          if (event.name === 'intent') setStage('intent', 'running');
          if (event.name === 'generate') setStage('generate', 'running');
          break;
        case 'plan':
          setStage('intent', 'done', event.elapsed);
          patchTrace({
            plan: {
              route: event.route,
              capabilities: event.capabilities,
              routeScores: event.routeScores,
              topic: event.topic,
              entities: event.entities,
              gate: event.gate,
              source: event.source,
              elapsed: event.elapsed,
            },
          });
          updateLast({
            route: event.route,
            domain: event.topic?.label || undefined,
          });
          break;
        case 'retrieve':
          setStage('retrieve', 'done', event.elapsed);
          patchTrace({ candidates: event.candidates });
          break;
        case 'rerank':
          setStage('rerank', 'done', event.elapsed);
          patchTrace({ selected: event.selected });
          break;
        case 'context':
          patchTrace({
            context: {
              sources: event.sources,
              conflict: event.conflict,
              notes: event.notes,
              chars: event.chars,
            },
          });
          // 订单工具超时/异常时不会发 order 事件，但 fan-out 已经结束：
          // 能力集里有订单却没收到结果，也要把这一步收口，避免停在「待执行」。
          setTrace((prev) => {
            const planned = (prev.plan?.capabilities ?? []).includes('order');
            if (!planned || prev.stages.order.status !== 'pending') return prev;
            return {
              ...prev,
              stages: { ...prev.stages, order: { status: 'done' } },
            };
          });
          break;
        case 'verdict':
          patchTrace({
            verdict: {
              verdict: event.verdict,
              reason: event.reason,
              score: event.score,
              thresholds: event.thresholds,
            },
          });
          break;
        case 'token':
          buffer += event.text;
          break;
        case 'reply':
          finish(event.text);
          // no_context 走固定话术，生成从未启动：不要把它标成已完成。
          setTrace((prev) =>
            prev.verdict?.verdict === 'no_context'
              ? { ...prev, active: false }
              : {
                  ...prev,
                  active: false,
                  stages: { ...prev.stages, generate: { status: 'done' } },
                }
          );
          break;
        case 'order':
          setStage('order', 'done', event.elapsed);
          pendingOrder = event.result;
          // 订单迟到时回答已经落地：直接补上订单卡。
          if (finished) updateLast({ order: event.result });
          break;
        case 'done':
          // reply 已定稿时 done 只收口，不再重写文本。
          if (finished) updateLast({ streaming: false });
          else finish(event.reply ?? '');
          setTrace((prev) =>
            prev.verdict?.verdict === 'no_context'
              ? { ...prev, active: false }
              : {
                  ...prev,
                  active: false,
                  stages: {
                    ...prev.stages,
                    generate: { status: 'done', elapsed: event.elapsed },
                  },
                }
          );
          break;
        case 'error':
          finish(buffer ? `${buffer}\n\n（${event.message}）` : event.message, true);
          patchTrace({ active: false });
          break;
      }
    };

    try {
      await streamChat(sessionId, query, userId, onEvent);
    } catch (e) {
      const detail = e instanceof Error ? e.message : String(e);
      finish(buffer ? `${buffer}\n\n（请求失败：${detail}）` : `请求失败：${detail}`, true);
      patchTrace({ active: false });
    } finally {
      // 流在 done 之前断开时，把已缓冲内容补上，避免留下空气泡。
      if (!finished && buffer) finish('');
      setMessages((prev) => prev.map((m) => ({ ...m, streaming: false })));
      setBusy(false);
    }
  };

  const newSession = async () => {
    await clearHistory(sessionId);
    setMessages([]);
    setTrace(emptyTrace());
    setSuggestionSeed(Date.now());
  };

  const switchUser = async (nextUserId: string) => {
    setUserId(nextUserId);
    setOrderIds([]);
    localStorage.setItem(USER_KEY, nextUserId);
    await newSession();
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">CS</span>
          <div>
            <h1>客服 Pilot</h1>
            <p>意图识别 → 工具/检索并行 → 流式生成 · 决策链路可观测</p>
          </div>
        </div>
        <div className="topbar-right">
          <label className="user-switcher">
            <span>当前用户</span>
            <select
              value={userId}
              onChange={(e) => switchUser(e.target.value)}
              disabled={busy}
            >
              {users.length === 0 && <option value={userId}>加载中…</option>}
              {users.map((u) => (
                <option key={u.userId} value={u.userId}>
                  {u.name} · {u.userId}
                </option>
              ))}
            </select>
          </label>
          <span
            className={`status-dot ${modelStatus === 'ready' ? 'ok' : 'warn'}`}
            aria-hidden
          />
          <span className="status-text">
            {modelStatus === 'ready'
              ? '模型就绪'
              : modelStatus === 'loading'
                ? '模型加载中…'
                : '模型服务未连接'}
          </span>
          <button className="btn-ghost" onClick={newSession}>
            新会话
          </button>
        </div>
      </header>

      <main className="layout">
        <ChatPanel
          messages={messages}
          busy={busy}
          suggestions={suggestions}
          bottomRef={bottomRef}
          onSend={send}
        />
        <TracePanel trace={trace} />
      </main>
    </div>
  );
}
