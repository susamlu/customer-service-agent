package com.example.csagent.chat;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.slf4j.MDC;
import org.springframework.stereotype.Component;

/**
 * 能力并行：同一轮里多个能力（订单查询、检索→重排）并发执行，各自独立超时（architecture.md §5.2）。
 *
 * <pre>
 *  提交全部任务           逐个等各自 deadline
 *  ┌──────────┐         ┌────────────────────────┐
 *  │ order    │──┐   ┌─▶│ get(300ms)   → 结果     │
 *  │ retrieve │──┼───┘  │ get(500ms)   → 结果     │
 *  └──────────┘         └────────────────────────┘
 *  慢任务只丢自己，不拖累其它能力
 * </pre>
 *
 * <p>超时两道保险：任务内的 HTTP 请求自带 timeout（真正结束 I/O），这里再用
 * {@code Future.get(timeout)} 兜住线程调度延迟。只等不抛：调用方按 {@link Outcome}
 * 决定降级，不让一个能力的失败冒泡成整轮失败。
 *
 * <p>线程池有界：任务全是阻塞网络 I/O，线程数给到核数的数倍；队列满时直接拒绝，
 * 被拒的能力收敛成失败 Outcome（这一轮降级），不排队等待——排队会把能力的超时预算
 * 白白耗在队列里，还会把上游雪崩传回来。
 */
@Component
public class FanOut {

    private static final int POOL_SIZE = Math.max(8, Runtime.getRuntime().availableProcessors() * 4);
    private static final int QUEUE_CAPACITY = 256;

    private final ExecutorService pool;

    public FanOut() {
        this(new ThreadPoolExecutor(
                POOL_SIZE,
                POOL_SIZE,
                0L,
                TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<>(QUEUE_CAPACITY),
                runnable -> {
                    Thread thread = new Thread(runnable, "csagent-fanout");
                    thread.setDaemon(true);
                    return thread;
                },
                new ThreadPoolExecutor.AbortPolicy()));
    }

    /** 包可见：测试注入已满 / 已关闭的池，验证拒绝被收敛成 Outcome 而不是抛出。 */
    FanOut(ExecutorService pool) {
        this.pool = pool;
    }

    /** 一个可并行能力：名字用于回查结果，timeout 从提交时刻开始算。 */
    public record Task(String name, Duration timeout, Callable<?> body) {}

    /**
     * 一次能力的执行结果。失败与超时都收敛成数据，不抛异常。
     *
     * <p>value 声明为 Object 而不是泛型：任务列表里各能力返回值类型不同，
     * 通配符在提交/等待阶段会把捕获类型搞乱。取用时由 {@link #valueAs} 显式收窄，
     * 比泛型更能表达"每个能力自己知道自己返回什么"。
     */
    public record Outcome(
            String name, Object value, Throwable error, double elapsedSec, boolean timedOut) {

        public boolean ok() {
            return error == null && !timedOut;
        }

        public <T> T valueAs(Class<T> type) {
            return type.cast(value);
        }
    }

    /**
     * 并发执行全部任务。先全部提交再逐个等待，保证任务真正重叠而不是串行累加超时。
     *
     * @return 与入参同序的结果列表
     */
    public List<Outcome> all(List<Task> tasks) {
        if (tasks.isEmpty()) {
            return List.of();
        }
        // 任务跑在别的线程，MDC（trace_id）默认不跟过去，提交时整份复制。
        Map<String, String> context = MDC.getCopyOfContextMap();
        List<InFlight> inFlight = new ArrayList<>(tasks.size());
        for (Task task : tasks) {
            long startedAt = System.nanoTime();
            Future<?> future;
            try {
                future = pool.submit(() -> callWithContext(context, task.body()));
            } catch (RejectedExecutionException e) {
                // 池满即降级：这一轮能力直接失败，调用方按既定逻辑回退。
                future = CompletableFuture.failedFuture(e);
            }
            inFlight.add(new InFlight(task, future, startedAt));
        }
        List<Outcome> outcomes = new ArrayList<>(inFlight.size());
        for (InFlight flight : inFlight) {
            outcomes.add(await(flight));
        }
        return outcomes;
    }

    /** 按任务名索引结果，调用方拿自己关心的能力即可。 */
    public Map<String, Outcome> byName(List<Outcome> outcomes) {
        Map<String, Outcome> byName = new LinkedHashMap<>();
        for (Outcome outcome : outcomes) {
            byName.put(outcome.name(), outcome);
        }
        return byName;
    }

    /** 在任务线程上挂载调用线程的 MDC，执行完恢复原状（线程池复用，不能残留）。 */
    private static <T> T callWithContext(Map<String, String> context, Callable<T> body)
            throws Exception {
        if (context == null || context.isEmpty()) {
            return body.call();
        }
        Map<String, String> previous = MDC.getCopyOfContextMap();
        MDC.setContextMap(context);
        try {
            return body.call();
        } finally {
            if (previous == null) {
                MDC.clear();
            } else {
                MDC.setContextMap(previous);
            }
        }
    }

    private Outcome await(InFlight flight) {
        long remainingNanos = flight.task().timeout().toNanos()
                - (System.nanoTime() - flight.startedAt());
        long remainingMillis = Math.max(0, TimeUnit.NANOSECONDS.toMillis(remainingNanos));
        try {
            Object value = flight.future().get(remainingMillis, TimeUnit.MILLISECONDS);
            return new Outcome(flight.task().name(), value, null, flight.elapsed(), false);
        } catch (TimeoutException e) {
            flight.future().cancel(true);
            return new Outcome(flight.task().name(), null, null, flight.elapsed(), true);
        } catch (ExecutionException e) {
            return new Outcome(
                    flight.task().name(), null, e.getCause(), flight.elapsed(), false);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            flight.future().cancel(true);
            return new Outcome(flight.task().name(), null, e, flight.elapsed(), true);
        }
    }

    private record InFlight(Task task, Future<?> future, long startedAt) {
        double elapsed() {
            return (System.nanoTime() - startedAt) / 1e9;
        }
    }
}
