export type StageName = 'intent' | 'order' | 'retrieve' | 'rerank' | 'generate';

export type StageStatus = 'pending' | 'running' | 'done';

export interface ScoredText {
  text: string;
  score: number;
  intent?: string;
}

/** 前置门控结果：为什么某个能力被保留或被裁掉（architecture.md §4.4）。 */
export interface GateDecision {
  allowed: boolean;
  reason: string;
}

/** /api/plan 的收敛结果：能力集决定通路，主题只在知识库通路里用到。 */
export interface PlanResult {
  route: string;
  capabilities: string[];
  routeScores: Record<string, number>;
  topic: { label: string; confidence: number };
  entities?: Record<string, string>;
  gate: Record<string, GateDecision>;
  source: string;
  elapsed: number;
}

/** 上下文拼装结果：本轮注入了哪些来源、是否发生订单/知识冲突（architecture.md §5.3）。 */
export interface ContextInfo {
  sources: string[];
  conflict: boolean;
  notes: string[];
  chars: number;
}

/** 可答性闸门三档判定（architecture.md §5.4）：没有可信知识时不调生成。 */
export interface VerdictResult {
  verdict: 'answer' | 'weak' | 'no_context';
  reason: string;
  score?: number;
  thresholds: { low: number; high: number };
}

export interface Trace {
  active: boolean;
  stages: Record<StageName, { status: StageStatus; elapsed?: number }>;
  plan?: PlanResult;
  context?: ContextInfo;
  verdict?: VerdictResult;
  candidates?: ScoredText[];
  selected?: ScoredText[];
}

export interface OrderResult {
  orderId: string;
  statusLabel: string;
  items: string;
  latestTrace: string;
  placedAt: string;
}

export interface Message {
  role: 'user' | 'assistant';
  text: string;
  route?: string;
  domain?: string;
  order?: OrderResult;
  streaming?: boolean;
  error?: boolean;
}

export type StreamEvent =
  | { type: 'stage'; name: StageName; status: 'start' }
  | {
      type: 'plan';
      route: string;
      capabilities: string[];
      routeScores: Record<string, number>;
      topic: { label: string; confidence: number };
      entities?: Record<string, string>;
      gate: Record<string, GateDecision>;
      source: string;
      elapsed: number;
    }
  | { type: 'retrieve'; candidates: ScoredText[]; elapsed: number }
  | { type: 'rerank'; selected: ScoredText[]; elapsed: number }
  | ({
      type: 'context';
    } & ContextInfo)
  | ({ type: 'verdict' } & VerdictResult)
  | { type: 'token'; text: string }
  | { type: 'reply'; text: string }
  | { type: 'order'; result: OrderResult; elapsed: number }
  | { type: 'done'; reply?: string; elapsed?: number }
  | { type: 'error'; code?: string; message: string };

export const emptyTrace = (): Trace => ({
  active: false,
  stages: {
    intent: { status: 'pending' },
    order: { status: 'pending' },
    retrieve: { status: 'pending' },
    rerank: { status: 'pending' },
    generate: { status: 'pending' },
  },
});
