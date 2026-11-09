package com.example.csagent.obs;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import java.time.Duration;
import org.springframework.stereotype.Component;

/**
 * 端到端链路指标（architecture.md §11）：每一段都用与 SSE 同源的分数计时，
 * 面板上看到的和监控里量到的是同一件事。
 *
 * <pre>
 *  cs.chat.plan{source,route}              L0/L1/L2 判定耗时 + 来源分布
 *  cs.chat.capability{capability,outcome}  order / retrieve / rerank 各自的耗时与降级
 *  cs.chat.generate{phase}                 first_token（首字延迟）/ total
 *  cs.chat.verdict{verdict,reason}         可答性闸门三档判定（§5.4）
 *  cs.chat.requests{outcome}               ok / error / disconnect
 * </pre>
 *
 * <p>直方图打开（{@code publishPercentileHistogram}）：Prometheus 端用
 * {@code histogram_quantile} 算 P95，与 README 的延迟门槛对齐。
 */
@Component
public class ChatMetrics {

    private static final String PLAN = "cs.chat.plan";
    private static final String CAPABILITY = "cs.chat.capability";
    private static final String GENERATE = "cs.chat.generate";
    private static final String VERDICT = "cs.chat.verdict";
    private static final String REQUESTS = "cs.chat.requests";

    private final MeterRegistry registry;

    public ChatMetrics(MeterRegistry registry) {
        this.registry = registry;
    }

    public void plan(Duration elapsed, String source, String route) {
        timer(PLAN, "能力规划耗时", "source", orUnknown(source), "route", orUnknown(route))
                .record(elapsed);
    }

    /** capability 取 {@code order|retrieve|rerank}，outcome 取 {@code ok|empty|degraded|missing|no_context}。 */
    public void capability(Duration elapsed, String capability, String outcome) {
        timer(
                        CAPABILITY,
                        "单能力耗时",
                        "capability",
                        orUnknown(capability),
                        "outcome",
                        orUnknown(outcome))
                .record(elapsed);
    }

    public void firstToken(Duration elapsed) {
        timer(GENERATE, "生成耗时", "phase", "first_token").record(elapsed);
    }

    public void generate(Duration elapsed) {
        timer(GENERATE, "生成耗时", "phase", "total").record(elapsed);
    }

    /** verdict 取 {@code answer|weak|no_context}，reason 取 {@code recall_score|empty|order_hit|order_only|degraded}。 */
    public void verdict(String verdict, String reason) {
        Counter.builder(VERDICT)
                .description("可答性闸门判定分布")
                .tag("verdict", orUnknown(verdict))
                .tag("reason", orUnknown(reason))
                .register(registry)
                .increment();
    }

    public void request(String outcome) {
        Counter.builder(REQUESTS)
                .description("一轮问答的最终结局")
                .tag("outcome", orUnknown(outcome))
                .register(registry)
                .increment();
    }

    /** 取（必要时注册）一个带直方图的计时器；tags 按 name/value 成对传入。 */
    private Timer timer(String name, String description, String... tags) {
        return Timer.builder(name)
                .description(description)
                .tags(tags)
                .publishPercentileHistogram()
                .register(registry);
    }

    private static String orUnknown(String value) {
        return value == null || value.isBlank() ? "unknown" : value;
    }
}
