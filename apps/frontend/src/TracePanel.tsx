import type {
  ContextInfo,
  PlanResult,
  ScoredText,
  StageName,
  Trace,
  VerdictResult,
} from './types';

const STAGE_META: { key: StageName; title: string; note: string }[] = [
  {
    key: 'intent',
    title: '意图识别',
    note: '多任务 BERT · 能力集 → 前置门控',
  },
  { key: 'order', title: '订单工具', note: '订单状态 / 物流 · 与知识召回并行' },
  { key: 'retrieve', title: '向量召回', note: 'bge-small-zh · 按主题分区加权 · top 10' },
  { key: 'rerank', title: '知识精选', note: '召回序 · top 2（默认不做精排）' },
  { key: 'generate', title: '生成回答', note: 'DeepSeek · 流式' },
];

const ROUTE_LABEL: Record<string, string> = {
  human: '转人工',
  order: '订单查询',
  knowledge: '知识库问答',
  clarify: '澄清追问',
};

const CAPABILITY_LABEL: Record<string, string> = {
  human: '人工',
  order: '订单',
  knowledge: '知识库',
};

const SOURCE_LABEL: Record<string, string> = {
  rule: '规则命中',
  bert: 'BERT 推理',
};

const VERDICT_LABEL: Record<string, string> = {
  answer: '直接生成',
  weak: '依据较弱',
  no_context: '无可靠依据',
};

const VERDICT_REASON: Record<string, string> = {
  recall_score: '召回首位分数',
  empty: '检索为空',
  order_hit: '订单事实兜底',
  order_only: '仅订单能力',
  degraded: '召回降级',
};

/** 统一耗时展示：内存订单查询是微秒级，直接按秒打印会出一串零。 */
function formatElapsed(seconds?: number): string {
  if (seconds === undefined) return '';
  if (seconds < 0.001) return '<1ms';
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  return `${seconds.toFixed(2)}s`;
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.max(4, Math.min(100, score * 100));
  return (
    <span className="score-bar">
      <span className={`score-track ${score >= 0.6 ? 'high' : ''}`}>
        <span className="score-fill" style={{ width: `${pct}%` }} />
      </span>
      <span className="score-num">{score.toFixed(3)}</span>
    </span>
  );
}

function CandidateList({ title, items }: { title: string; items: ScoredText[] }) {
  if (!items?.length) return null;
  return (
    <div className="candidates">
      <div className="candidates-title">{title}</div>
      {items.map((item, i) => (
        <div key={i} className="candidate">
          <span className="candidate-text">{item.text}</span>
          <span className="candidate-side">
            {item.intent && <span className="candidate-intent">{item.intent}</span>}
            <ScoreBar score={item.score} />
          </span>
        </div>
      ))}
    </div>
  );
}

function PlanBlock({ plan }: { plan: PlanResult }) {
  const gates = Object.entries(plan.gate ?? {});
  return (
    <div className="trace-block">
      <div className="block-title">意图识别</div>
      <div className={`tool-chip ${plan.route}`}>
        {ROUTE_LABEL[plan.route] ?? plan.route}
      </div>
      {plan.capabilities.length > 0 && (
        <div className="route-tries">
          {plan.capabilities.map((name) => (
            <span key={name} className="route-try hit">
              {CAPABILITY_LABEL[name] ?? name}
            </span>
          ))}
        </div>
      )}
      {plan.topic.label && <div className="intent-chip">主题 · {plan.topic.label}</div>}
      {plan.topic.confidence > 0 && (
        <div className="block-quiet">主题置信度 {plan.topic.confidence.toFixed(3)}</div>
      )}
      {plan.entities?.orderId && (
        <div className="block-quiet">实体 · 订单 {plan.entities.orderId}</div>
      )}
      {plan.source && (
        <div className="block-quiet">来源 {SOURCE_LABEL[plan.source] ?? plan.source}</div>
      )}
      {gates.length > 0 && (
        <div className="gate-list">
          {gates.map(([name, decision]) => (
            <span key={name} className={`gate ${decision.allowed ? 'ok' : 'blocked'}`}>
              {CAPABILITY_LABEL[name] ?? name} · {decision.allowed ? '通过' : '裁掉'}
              {!decision.allowed && decision.reason ? `（${decision.reason}）` : ''}
            </span>
          ))}
        </div>
      )}
      {plan.elapsed > 0 && (
        <div className="block-quiet">耗时 {formatElapsed(plan.elapsed)}</div>
      )}
    </div>
  );
}

/** 上下文拼装结果：哪些来源进了 prompt、是否发生时效冲突。 */
function ContextBlock({ context }: { context: ContextInfo }) {
  return (
    <div className="trace-block">
      <div className="block-title">上下文拼装</div>
      {context.sources.length === 0 ? (
        <div className="block-quiet">无可用上下文，按常识回答</div>
      ) : (
        <div className="gate-list">
          {context.sources.map((source) => (
            <span key={source} className="gate ok">
              {CAPABILITY_LABEL[source] ?? source}
            </span>
          ))}
          {context.conflict && (
            <span className="gate blocked">时效冲突 · 以订单承诺为准</span>
          )}
        </div>
      )}
      {context.notes.map((note) => (
        <div key={note} className="block-quiet">
          {note}
        </div>
      ))}
      <div className="block-quiet">{context.chars} 字注入生成</div>
    </div>
  );
}

/** 可答性闸门：三档判定 + 依据 + 阈值，解释「这一轮为什么生成 / 为什么没生成」。 */
function VerdictBlock({ verdict }: { verdict: VerdictResult }) {
  const { low, high } = verdict.thresholds;
  return (
    <div className="trace-block">
      <div className="block-title">可答性判定</div>
      <div className={`tool-chip ${verdict.verdict}`}>
        {VERDICT_LABEL[verdict.verdict] ?? verdict.verdict}
      </div>
      <div className="block-quiet">
        依据 {VERDICT_REASON[verdict.reason] ?? verdict.reason}
        {verdict.score !== undefined ? ` · 首位分 ${verdict.score.toFixed(3)}` : ''}
      </div>
      <div className="block-quiet">
        阈值 low {low.toFixed(2)} / high {high.toFixed(2)}
      </div>
      {verdict.verdict === 'no_context' && (
        <div className="block-quiet">知识库无可靠依据，未调用生成模型</div>
      )}
      {verdict.verdict === 'weak' && (
        <div className="block-quiet">照常生成，依据较弱</div>
      )}
    </div>
  );
}

export default function TracePanel({ trace }: { trace: Trace }) {
  const idle = !trace.active && !trace.plan;
  // 能力集可能同时命中订单和知识：每个环节是否相关看能力集，而不是被展示的单一 route。
  const capabilities = trace.plan?.capabilities ?? [];
  const knowledge = capabilities.includes('knowledge');
  const order = capabilities.includes('order');
  return (
    <aside className="trace">
      <div className="trace-head">
        <h2>管线追踪</h2>
        <p>{idle ? '发送一个问题，看看链路如何工作' : '实时'}</p>
      </div>
      <ol className="stages">
        {STAGE_META.map((meta) => {
          const stage = trace.stages[meta.key];
          // 空闲时展示完整管线说明，问过一次后再按能力集标记「本分支未经过」。
          const relevant =
            idle ||
            meta.key === 'intent' ||
            meta.key === 'generate' ||
            (meta.key === 'order' && order) ||
            ((meta.key === 'retrieve' || meta.key === 'rerank') && knowledge);
          return (
            <li
              key={meta.key}
              className={`stage ${stage.status} ${relevant ? '' : 'skipped'}`}
            >
              <div className="stage-top">
                <span className="stage-dot" aria-hidden />
                <span className="stage-title">{meta.title}</span>
                {stage.status === 'running' && <span className="stage-running">运行中</span>}
                {stage.status === 'done' && stage.elapsed !== undefined && (
                  <span className="stage-elapsed">{formatElapsed(stage.elapsed)}</span>
                )}
              </div>
              <div className="stage-note">{relevant ? meta.note : '本分支未经过'}</div>
            </li>
          );
        })}
      </ol>

      {trace.plan && <PlanBlock plan={trace.plan} />}
      {trace.context && <ContextBlock context={trace.context} />}
      {trace.verdict && <VerdictBlock verdict={trace.verdict} />}

      <CandidateList title="送入生成的知识（top 2）" items={trace.selected ?? []} />
      <CandidateList title="召回候选 top 5" items={trace.candidates ?? []} />
    </aside>
  );
}
