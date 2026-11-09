package com.example.csagent.chat;

import static org.assertj.core.api.Assertions.assertThat;

import com.example.csagent.chat.AnswerabilityGate.Decision;
import com.example.csagent.chat.AnswerabilityGate.Reason;
import com.example.csagent.chat.AnswerabilityGate.Verdict;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** 可答性闸门决策表（§5.4）：每一行分支都要有断言，阈值改动必须和评测脚本同步。 */
class AnswerabilityGateTest {

    @Test
    @DisplayName("thresholds stay in sync with eval_answerability.py")
    void thresholdsMatchEval() {
        assertThat(AnswerabilityGate.T_LOW).isEqualTo(0.58);
        assertThat(AnswerabilityGate.T_HIGH).isEqualTo(0.64);
    }

    @Test
    @DisplayName("score at or above T_HIGH answers from recall")
    void highScoreAnswers() {
        Decision decision = AnswerabilityGate.decide(true, false, false, AnswerabilityGate.T_HIGH);

        assertThat(decision.verdict()).isEqualTo(Verdict.ANSWER);
        assertThat(decision.reason()).isEqualTo(Reason.RECALL_SCORE);
        assertThat(decision.score()).isEqualTo(AnswerabilityGate.T_HIGH);
    }

    @Test
    @DisplayName("score between T_LOW and T_HIGH is weak")
    void middleScoreIsWeak() {
        Decision decision = AnswerabilityGate.decide(true, false, false, AnswerabilityGate.T_LOW);

        assertThat(decision.verdict()).isEqualTo(Verdict.WEAK);
        assertThat(decision.reason()).isEqualTo(Reason.RECALL_SCORE);
    }

    @Test
    @DisplayName("score below T_LOW with no order is no_context")
    void lowScoreWithoutOrderIsNoContext() {
        Decision decision =
                AnswerabilityGate.decide(true, false, false, AnswerabilityGate.T_LOW - 0.01);

        assertThat(decision.verdict()).isEqualTo(Verdict.NO_CONTEXT);
        assertThat(decision.reason()).isEqualTo(Reason.RECALL_SCORE);
    }

    @Test
    @DisplayName("score below T_LOW with order facts never blocks generation")
    void lowScoreWithOrderIsWeak() {
        Decision decision =
                AnswerabilityGate.decide(true, false, true, AnswerabilityGate.T_LOW - 0.01);

        assertThat(decision.verdict()).isEqualTo(Verdict.WEAK);
        assertThat(decision.reason()).isEqualTo(Reason.ORDER_HIT);
    }

    @Test
    @DisplayName("empty recall with order is weak, without order is no_context")
    void emptyRecall() {
        assertThat(AnswerabilityGate.decide(true, false, true, null).verdict())
                .isEqualTo(Verdict.WEAK);
        assertThat(AnswerabilityGate.decide(true, false, true, null).reason())
                .isEqualTo(Reason.ORDER_HIT);

        Decision noContext = AnswerabilityGate.decide(true, false, false, null);
        assertThat(noContext.verdict()).isEqualTo(Verdict.NO_CONTEXT);
        assertThat(noContext.reason()).isEqualTo(Reason.EMPTY);
        assertThat(noContext.score()).isNull();
    }

    @Test
    @DisplayName("degraded recall trusts order facts, otherwise falls back to weak")
    void degradedRecall() {
        assertThat(AnswerabilityGate.decide(true, true, true, null).verdict())
                .isEqualTo(Verdict.ANSWER);
        assertThat(AnswerabilityGate.decide(true, true, true, null).reason())
                .isEqualTo(Reason.ORDER_HIT);

        Decision degraded = AnswerabilityGate.decide(true, true, false, null);
        assertThat(degraded.verdict()).isEqualTo(Verdict.WEAK);
        assertThat(degraded.reason()).isEqualTo(Reason.DEGRADED);
    }

    @Test
    @DisplayName("order-only plan answers when order found, weak otherwise")
    void orderOnlyPlan() {
        Decision found = AnswerabilityGate.decide(false, false, true, null);
        assertThat(found.verdict()).isEqualTo(Verdict.ANSWER);
        assertThat(found.reason()).isEqualTo(Reason.ORDER_ONLY);

        Decision none = AnswerabilityGate.decide(false, false, false, null);
        assertThat(none.verdict()).isEqualTo(Verdict.WEAK);
        assertThat(none.reason()).isEqualTo(Reason.ORDER_ONLY);
    }

    @Test
    @DisplayName("no_context reply never tells the user we guessed")
    void replyText() {
        assertThat(AnswerabilityGate.NO_CONTEXT_REPLY).doesNotContain("猜测");
        assertThat(AnswerabilityGate.NO_CONTEXT_REPLY).contains("转人工");
    }
}
