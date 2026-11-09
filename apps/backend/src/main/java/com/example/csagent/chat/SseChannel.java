package com.example.csagent.chat;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * SSE 输出通道（architecture.md §11）：事件封装 + 心跳 + 断连标记。
 *
 * <p>所有写操作串行化：心跳线程与编排线程共用同一个 {@link OutputStream}，
 * 不加锁就可能把一个 {@code data:} 帧劈成两半，前端解析直接报错。
 *
 * <p>心跳用 SSE 注释行（{@code : ping}），客户端不渲染但能让中间代理看到流量，
 * 避免首字前的长思考或上游卡住时被 60s 空闲超时掐断。
 *
 * <p>心跳跑在共享的调度池里，而不是单线程：{@code OutputStream.write} 对端不读时会阻塞，
 * 单线程下只要有一个慢客户端卡住，其他所有连接的心跳都会被一起饿死，然后被代理掐断。
 * 多线程把风险从"一个慢客户端拖死全部"降到"多个"，彻底解决要靠容器侧的写超时。
 *
 * <p>写失败即视为客户端断开：置 {@code closed} 并抛 {@link UncheckedIOException}，
 * 编排器据此停止生成、关闭上游连接（§8「客户端断开 → 取消上游请求」）。
 */
public final class SseChannel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(SseChannel.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final long HEARTBEAT_MILLIS = TimeUnit.SECONDS.toMillis(15);
    private static final int HEARTBEAT_POOL_SIZE = 4;
    private static final ScheduledExecutorService HEARTBEAT_EXECUTOR =
            Executors.newScheduledThreadPool(HEARTBEAT_POOL_SIZE, runnable -> {
                Thread thread = new Thread(runnable, "sse-heartbeat");
                thread.setDaemon(true);
                return thread;
            });

    private final OutputStream out;
    private final Object lock = new Object();
    private final ScheduledFuture<?> heartbeat;
    private volatile boolean closed;

    private SseChannel(
            OutputStream out, ScheduledExecutorService heartbeatExecutor, long heartbeatMillis) {
        this.out = out;
        this.heartbeat = heartbeatExecutor.scheduleAtFixedRate(
                this::ping, heartbeatMillis, heartbeatMillis, TimeUnit.MILLISECONDS);
    }

    public static SseChannel open(OutputStream out) {
        return new SseChannel(out, HEARTBEAT_EXECUTOR, HEARTBEAT_MILLIS);
    }

    /** 包可见：测试用短间隔 + 指定的共享池，复现"一个慢连接不拖垮其他连接的心跳"。 */
    static SseChannel open(
            OutputStream out, ScheduledExecutorService heartbeatExecutor, long heartbeatMillis) {
        return new SseChannel(out, heartbeatExecutor, heartbeatMillis);
    }

    /** 写一条 JSON 事件：{@code data: {...}\n\n}。 */
    public void send(Map<String, ?> payload) {
        try {
            write("data: " + MAPPER.writeValueAsString(payload) + "\n\n");
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("SSE 事件序列化失败", e);
        }
    }

    /** 原样转发上游 SSE 行（上游的 data: 前缀与空行由它自己带）。 */
    public void sendLine(String line) {
        write(line + "\n");
    }

    public boolean isClosed() {
        return closed;
    }

    private void ping() {
        if (closed) {
            return;
        }
        try {
            write(": ping\n\n");
        } catch (UncheckedIOException e) {
            // 客户端已走，通道已置 closed；定时任务本身不能抛，否则线程会被调度器丢弃。
            // 顺手取消自己，省掉后续每秒一次的空转。
            heartbeat.cancel(false);
            log.debug("heartbeat stopped: client disconnected");
        }
    }

    private void write(String text) {
        synchronized (lock) {
            if (closed) {
                throw new UncheckedIOException(
                        "客户端断开连接", new IOException("channel closed"));
            }
            try {
                out.write(text.getBytes(StandardCharsets.UTF_8));
                out.flush();
            } catch (IOException e) {
                closed = true;
                throw new UncheckedIOException("客户端断开连接", e);
            }
        }
    }

    @Override
    public void close() {
        closed = true;
        heartbeat.cancel(false);
    }
}
