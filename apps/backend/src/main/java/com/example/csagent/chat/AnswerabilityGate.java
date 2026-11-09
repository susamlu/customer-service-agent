package com.example.csagent.chat;

/**
 * 可答性闸门（architecture.md §5.4）：用召回首位分数把知识能力判成
 * answer / weak / no_context 三档。
 *
 * <pre>
 *   score >= T_HIGH           → answer      直接生成
 *   T_LOW <= score < T_HIGH   → weak        照常生成，前端提示依据较弱
 *   score < T_LOW             → no_context  走固定话术，不调生成模型
 *   检索为空                   → no_context
 *   召回异常（降级）            → weak        数据不足，不拦
 *   只有订单能力               → answer      order_only
 *   有订单事实且知识弱          → weak        绝不 no_context
 * </pre>
 *
 * <p>阈值只用在召回首位分数上。微调 reranker 的绝对分数没有校准能力（域外
 * 问题同样打 0.9+），不能当阈值，详见 §5.4「信号选型」。
 *
 * <p>产品取向「宁可漏拦，不可误拦」：只有既没有可信知识上下文、也没有订单
 * 事实时才拦截生成——把可答问题变成固定话术的代价高于偶尔答一条无关内容。
 */
public final class AnswerabilityGate {

    /** 与 ml/experiments/eval_answerability.py 的 T_LOW / T_HIGH 同值，改动必须两边同步。 */
    public static final double T_LOW = 0.58;
    public static final double T_HIGH = 0.64;

    /** no_context 的固定话术：宁可说没查到，也不让模型用常识补。 */
    public static final String NO_CONTEXT_REPLY =
            "这个问题我暂时没有查到可靠的相关说明，不想给你不准确的信息。"
                    + "你可以补充更多细节，或回复\"转人工\"由客服协助处理。";

    public enum Verdict { ANSWER, WEAK, NO_CONTEXT }

    public enum Reason { RECALL_SCORE, EMPTY, ORDER_HIT, ORDER_ONLY, DEGRADED }

    public record Decision(Verdict verdict, Reason reason, Double score) {}

    private AnswerabilityGate() {}

    /**
     * @param knowledgePlanned  本轮能力集是否包含 knowledge
     * @param retrievalDegraded 召回能力超时/异常（此时没有可信分数）
     * @param orderFound        本轮是否拿到了订单事实
     * @param topScore          召回首位分数；无召回时为 {@code null}
     */
    public static Decision decide(
            boolean knowledgePlanned,
            boolean retrievalDegraded,
            boolean orderFound,
            Double topScore) {
        if (!knowledgePlanned) {
            // 订单能力已经在调用方处理 Missing / NoContext，走到这里说明能答。
            return orderFound
                    ? new Decision(Verdict.ANSWER, Reason.ORDER_ONLY, null)
                    : new Decision(Verdict.WEAK, Reason.ORDER_ONLY, null);
        }
        if (retrievalDegraded) {
            return orderFound
                    ? new Decision(Verdict.ANSWER, Reason.ORDER_HIT, null)
                    : new Decision(Verdict.WEAK, Reason.DEGRADED, null);
        }
        if (topScore == null) {
            return orderFound
                    ? new Decision(Verdict.WEAK, Reason.ORDER_HIT, null)
                    : new Decision(Verdict.NO_CONTEXT, Reason.EMPTY, null);
        }
        if (topScore >= T_HIGH) {
            return new Decision(Verdict.ANSWER, Reason.RECALL_SCORE, topScore);
        }
        if (topScore >= T_LOW) {
            return new Decision(Verdict.WEAK, Reason.RECALL_SCORE, topScore);
        }
        return orderFound
                ? new Decision(Verdict.WEAK, Reason.ORDER_HIT, topScore)
                : new Decision(Verdict.NO_CONTEXT, Reason.RECALL_SCORE, topScore);
    }
}
