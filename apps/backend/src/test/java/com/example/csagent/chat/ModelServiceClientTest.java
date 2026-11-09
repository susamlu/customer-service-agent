package com.example.csagent.chat;

import static org.assertj.core.api.Assertions.assertThat;

import com.example.csagent.obs.Trace;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.MDC;

/** 出站 trace_id（§11）：有请求上下文就带，没有就不带，且绝不影响调用。 */
class ModelServiceClientTest {

    private HttpServer server;
    private ModelServiceClient client;
    private final List<String> seenTraceHeaders = new CopyOnWriteArrayList<>();

    @BeforeEach
    void startStub() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            seenTraceHeaders.add(exchange.getRequestHeaders().getFirst(Trace.HEADER));
            String path = exchange.getRequestURI().getPath();
            byte[] body;
            if ("/api/generate".equals(path)) {
                body = "data: {\"type\":\"done\"}\n\n".getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().add("Content-Type", "text/event-stream");
            } else if ("/healthz".equals(path)) {
                body = "{\"status\":\"ready\"}".getBytes(StandardCharsets.UTF_8);
            } else {
                body = "{}".getBytes(StandardCharsets.UTF_8);
            }
            exchange.sendResponseHeaders(200, body.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(body);
            }
        });
        server.start();
        client = new ModelServiceClient(
                "http://127.0.0.1:" + server.getAddress().getPort());
    }

    @AfterEach
    void stopStub() {
        MDC.clear();
        server.stop(0);
    }

    @Test
    @DisplayName("plan/retrieve/rerank/generate/health all carry the current trace id")
    void everyCallCarriesTraceId() {
        MDC.put(Trace.MDC_KEY, "trace-abc");

        client.plan("退款多久到账", "s1", List.of());
        client.retrieve("退款多久到账", 5, "knowledge");
        client.rerank("退款多久到账", List.of("a", "b"), 2);
        try (Stream<String> lines =
                client.generateStream(List.of(Map.of("role", "user", "content", "hi")), 32)) {
            assertThat(lines.filter(line -> !line.isBlank()).toList())
                    .containsExactly("data: {\"type\":\"done\"}");
        }
        client.health();

        assertThat(seenTraceHeaders).containsExactly(
                "trace-abc", "trace-abc", "trace-abc", "trace-abc", "trace-abc");
    }

    @Test
    @DisplayName("no trace context sends no header and still succeeds")
    void noContextNoHeader() {
        client.plan("退款多久到账", "s1", List.of());

        assertThat(seenTraceHeaders).containsExactly((String) null);
    }
}
