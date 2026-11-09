package com.example.csagent.chat;

import static org.assertj.core.api.Assertions.assertThat;

import com.example.csagent.obs.Trace;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.MDC;

/** 能力并行（§5.2）：真并发、各自超时、失败收敛成数据。 */
class FanOutTest {

    private final FanOut fanOut = new FanOut();

    @AfterEach
    void clearMdc() {
        MDC.clear();
    }

    @Test
    @DisplayName("tasks run concurrently, not one after another")
    void tasksRunConcurrently() {
        // 两个任务互相等待对方启动：串行执行时第一个任务等不到信号，
        // 会先撞上自己的 1s 超时（timedOut=true），测试必然失败。
        CountDownLatch started = new CountDownLatch(2);
        FanOut.Task a = new FanOut.Task("a", Duration.ofSeconds(1), () -> {
            started.countDown();
            boolean both = started.await(2, TimeUnit.SECONDS);
            return "a:" + both;
        });
        FanOut.Task b = new FanOut.Task("b", Duration.ofSeconds(1), () -> {
            started.countDown();
            boolean both = started.await(2, TimeUnit.SECONDS);
            return "b:" + both;
        });

        Map<String, FanOut.Outcome> results = fanOut.byName(fanOut.all(List.of(a, b)));

        assertThat(results.get("a").ok()).isTrue();
        assertThat(results.get("b").ok()).isTrue();
        assertThat(results.get("a").valueAs(String.class)).isEqualTo("a:true");
        assertThat(results.get("b").valueAs(String.class)).isEqualTo("b:true");
    }

    @Test
    @DisplayName("slow capability times out without dragging the fast one down")
    void slowTaskTimesOutAlone() {
        FanOut.Task fast = new FanOut.Task("fast", Duration.ofSeconds(1), () -> "fast");
        FanOut.Task slow = new FanOut.Task("slow", Duration.ofMillis(50), () -> {
            Thread.sleep(400);
            return "slow";
        });

        Map<String, FanOut.Outcome> results = fanOut.byName(fanOut.all(List.of(fast, slow)));

        assertThat(results.get("fast").ok()).isTrue();
        assertThat(results.get("fast").valueAs(String.class)).isEqualTo("fast");
        assertThat(results.get("slow").ok()).isFalse();
        assertThat(results.get("slow").timedOut()).isTrue();
        assertThat(results.get("slow").error()).isNull();
    }

    @Test
    @DisplayName("exceptions become outcomes, never bubble to the caller")
    void exceptionsAreCaptured() {
        FanOut.Task failing = new FanOut.Task("failing", Duration.ofSeconds(1), () -> {
            throw new IllegalStateException("boom");
        });
        FanOut.Task fast = new FanOut.Task("fast", Duration.ofSeconds(1), () -> "fast");

        Map<String, FanOut.Outcome> results = fanOut.byName(fanOut.all(List.of(failing, fast)));

        assertThat(results.get("failing").ok()).isFalse();
        assertThat(results.get("failing").timedOut()).isFalse();
        assertThat(results.get("failing").error()).isInstanceOf(IllegalStateException.class);
        assertThat(results.get("fast").ok()).isTrue();
    }

    @Test
    @DisplayName("empty task list yields empty outcomes")
    void emptyTasks() {
        assertThat(fanOut.all(List.of())).isEmpty();
    }

    @Test
    @DisplayName("MDC (trace_id) follows tasks into the pool")
    void mdcFollowsTasks() {
        MDC.put(Trace.MDC_KEY, "trace-abc");

        List<FanOut.Task> tasks = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            tasks.add(new FanOut.Task("t" + i, Duration.ofSeconds(1), () -> MDC.get(Trace.MDC_KEY)));
        }

        assertThat(fanOut.all(tasks))
                .extracting(outcome -> outcome.valueAs(String.class))
                .containsOnly("trace-abc");
    }

    @Test
    @DisplayName("MDC does not leak into later runs on reused pool threads")
    void mdcDoesNotLeak() {
        MDC.put(Trace.MDC_KEY, "trace-abc");
        List<FanOut.Task> tagged = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            tagged.add(new FanOut.Task("t" + i, Duration.ofSeconds(1), () -> MDC.get(Trace.MDC_KEY)));
        }
        fanOut.all(tagged);
        MDC.clear();

        List<FanOut.Task> untagged = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            untagged.add(new FanOut.Task("u" + i, Duration.ofSeconds(1), () -> MDC.get(Trace.MDC_KEY)));
        }

        assertThat(fanOut.all(untagged))
                .extracting(outcome -> outcome.valueAs(String.class))
                .containsOnlyNulls();
    }

    @Test
    @DisplayName("a saturated pool rejects instead of queueing; rejection becomes a failed outcome")
    void saturatedPoolDegradesInsteadOfQueueing() {
        // 单线程 + SynchronousQueue：没有任何缓冲。第一个任务占住唯一的 worker，
        // 第二个任务提交时 offer 无处可去、线程数已到上限，拒绝是确定性的，不依赖时序。
        // 这正是要验证的语义——池满即降级，而不是排队等超时预算被白耗。
        ExecutorService singleSlot = new ThreadPoolExecutor(
                1, 1, 0L, TimeUnit.MILLISECONDS, new SynchronousQueue<>(),
                new ThreadPoolExecutor.AbortPolicy());
        CountDownLatch release = new CountDownLatch(1);
        FanOut boundedFanOut = new FanOut(singleSlot);
        try {
            FanOut.Task occupying = new FanOut.Task("occupying", Duration.ofMillis(200), () -> {
                release.await(2, TimeUnit.SECONDS);
                return "occupying";
            });
            FanOut.Task rejected = new FanOut.Task("rejected", Duration.ofSeconds(1), () -> "rejected");

            Map<String, FanOut.Outcome> results =
                    boundedFanOut.byName(boundedFanOut.all(List.of(occupying, rejected)));

            assertThat(results.get("occupying").ok()).isFalse();
            assertThat(results.get("occupying").timedOut()).isTrue();
            assertThat(results.get("rejected").ok()).isFalse();
            assertThat(results.get("rejected").error())
                    .isInstanceOf(RejectedExecutionException.class);
        } finally {
            release.countDown();
            singleSlot.shutdownNow();
        }
    }

    @Test
    @DisplayName("submitting to a shut-down pool yields a failed outcome, not an exception")
    void shutDownPoolYieldsFailedOutcome() {
        ExecutorService closed = Executors.newSingleThreadExecutor();
        closed.shutdown();
        FanOut closedFanOut = new FanOut(closed);

        List<FanOut.Outcome> outcomes =
                closedFanOut.all(List.of(new FanOut.Task("t", Duration.ofSeconds(1), () -> "x")));

        assertThat(outcomes).hasSize(1);
        assertThat(outcomes.get(0).ok()).isFalse();
        assertThat(outcomes.get(0).timedOut()).isFalse();
        assertThat(outcomes.get(0).error()).isInstanceOf(RejectedExecutionException.class);
    }
}
