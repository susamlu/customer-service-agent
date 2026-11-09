package com.example.csagent.chat;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** SSE 通道（§4）：帧格式、关闭语义、写失败即断连、心跳不互相饿死。 */
class SseChannelTest {

    @Test
    @DisplayName("send writes a data frame terminated by a blank line")
    void send_writesDataFrame() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (SseChannel channel = SseChannel.open(out)) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "done");
            channel.send(payload);
        }

        String written = out.toString(StandardCharsets.UTF_8);
        assertThat(written).startsWith("data: {").endsWith("}\n\n").contains("\"type\":\"done\"");
    }

    @Test
    @DisplayName("sendLine forwards the upstream line verbatim")
    void sendLine_appendsNewline() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (SseChannel channel = SseChannel.open(out)) {
            channel.sendLine("data: {\"type\":\"token\"}");
        }

        assertThat(out.toString(StandardCharsets.UTF_8))
                .isEqualTo("data: {\"type\":\"token\"}\n");
    }

    @Test
    @DisplayName("writing after close throws UncheckedIOException")
    void send_afterClose_throws() {
        SseChannel channel = SseChannel.open(new ByteArrayOutputStream());
        channel.close();

        assertThat(channel.isClosed()).isTrue();
        assertThatThrownBy(() -> channel.send(Map.of("type", "done")))
                .isInstanceOf(UncheckedIOException.class);
    }

    @Test
    @DisplayName("a failing stream marks the channel closed")
    void send_writeFailure_closesChannel() {
        OutputStream broken = new OutputStream() {
            @Override
            public void write(int b) throws IOException {
                throw new IOException("client gone");
            }
        };
        SseChannel channel = SseChannel.open(broken);

        assertThatThrownBy(() -> channel.send(Map.of("type", "done")))
                .isInstanceOf(UncheckedIOException.class);
        assertThat(channel.isClosed()).isTrue();
        channel.close();
    }

    @Test
    @DisplayName("heartbeat writes a ping comment at the configured interval")
    void heartbeat_writesPing() throws InterruptedException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
        SseChannel channel = SseChannel.open(out, scheduler, 20);
        try {
            await(() -> out.toString(StandardCharsets.UTF_8).contains(": ping\n\n"));
        } finally {
            channel.close();
            scheduler.shutdownNow();
        }
    }

    @Test
    @DisplayName("a blocked client does not starve other channels' heartbeats")
    void blockedClient_doesNotStarveOthers() throws InterruptedException {
        // 这个流模拟"客户端不读、socket buffer 满了"：write 永久阻塞直到放行。
        CountDownLatch blocked = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        OutputStream stuck = new OutputStream() {
            @Override
            public void write(int b) {
                blocked.countDown();
                try {
                    release.await(5, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        };
        // 两个线程的池：慢连接占住一个，健康连接仍有线程可用。单线程池下这个断言必然超时。
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);
        ByteArrayOutputStream healthy = new ByteArrayOutputStream();
        SseChannel slow = SseChannel.open(stuck, scheduler, 20);
        SseChannel fast = SseChannel.open(healthy, scheduler, 20);
        try {
            assertThat(blocked.await(2, TimeUnit.SECONDS)).isTrue();
            await(() -> healthy.toString(StandardCharsets.UTF_8).contains(": ping\n\n"));
        } finally {
            release.countDown();
            slow.close();
            fast.close();
            scheduler.shutdownNow();
        }
    }

    private static void await(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(10);
        }
        throw new AssertionError("condition not met within 2s");
    }
}
