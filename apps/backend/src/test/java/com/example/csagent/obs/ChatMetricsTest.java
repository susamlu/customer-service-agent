package com.example.csagent.obs;

import static org.assertj.core.api.Assertions.assertThat;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** 指标命名与标签：面板/告警依赖这些字符串，改动要在这里同步（§11）。 */
class ChatMetricsTest {

    private final SimpleMeterRegistry registry = new SimpleMeterRegistry();
    private final ChatMetrics metrics = new ChatMetrics(registry);

    @Test
    @DisplayName("plan measures duration and splits by source/route")
    void planTimer() {
        metrics.plan(Duration.ofMillis(12), "L1", "knowledge");
        metrics.plan(Duration.ofMillis(3), "L0", "order");

        assertThat(count("cs.chat.plan", "source", "L1", "route", "knowledge")).isEqualTo(1L);
        assertThat(maxMillis("cs.chat.plan", "source", "L1", "route", "knowledge")).isEqualTo(12.0);
        assertThat(count("cs.chat.plan", "source", "L0", "route", "order")).isEqualTo(1L);
    }

    @Test
    @DisplayName("capability measures duration and outcome")
    void capabilityTimer() {
        metrics.capability(Duration.ofMillis(300), "retrieve", "degraded");

        assertThat(count("cs.chat.capability", "capability", "retrieve", "outcome", "degraded"))
                .isEqualTo(1L);
    }

    @Test
    @DisplayName("generate splits first_token from total")
    void generatePhases() {
        metrics.firstToken(Duration.ofMillis(860));
        metrics.generate(Duration.ofMillis(1180));

        assertThat(count("cs.chat.generate", "phase", "first_token")).isEqualTo(1L);
        assertThat(count("cs.chat.generate", "phase", "total")).isEqualTo(1L);
    }

    @Test
    @DisplayName("request outcome counter splits ok/error/disconnect")
    void requestCounter() {
        metrics.request("ok");
        metrics.request("ok");
        metrics.request("error");
        metrics.request("disconnect");

        assertThat(counter("cs.chat.requests", "outcome", "ok")).isEqualTo(2.0);
        assertThat(counter("cs.chat.requests", "outcome", "error")).isEqualTo(1.0);
        assertThat(counter("cs.chat.requests", "outcome", "disconnect")).isEqualTo(1.0);
    }

    @Test
    @DisplayName("verdict counter splits by verdict and reason")
    void verdictCounter() {
        metrics.verdict("answer", "recall_score");
        metrics.verdict("answer", "recall_score");
        metrics.verdict("no_context", "empty");

        assertThat(counter("cs.chat.verdict", "verdict", "answer", "reason", "recall_score"))
                .isEqualTo(2.0);
        assertThat(counter("cs.chat.verdict", "verdict", "no_context", "reason", "empty"))
                .isEqualTo(1.0);
    }

    @Test
    @DisplayName("null or blank tags collapse to unknown, never create a new label")
    void blankTagsBecomeUnknown() {
        metrics.plan(Duration.ofMillis(1), null, " ");
        metrics.request(null);

        assertThat(count("cs.chat.plan", "source", "unknown", "route", "unknown")).isEqualTo(1L);
        assertThat(counter("cs.chat.requests", "outcome", "unknown")).isEqualTo(1.0);
    }

    private long count(String name, String... tags) {
        return registry.get(name).tags(tags).timer().count();
    }

    private double maxMillis(String name, String... tags) {
        return registry.get(name).tags(tags).timer().max(TimeUnit.MILLISECONDS);
    }

    private double counter(String name, String... tags) {
        return registry.get(name).tags(tags).counter().count();
    }
}
