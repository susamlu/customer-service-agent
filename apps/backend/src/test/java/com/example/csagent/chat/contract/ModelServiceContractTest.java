package com.example.csagent.chat.contract;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.example.csagent.chat.CapabilityPlanner;
import com.example.csagent.chat.ChatMessage;
import com.example.csagent.chat.ErrorCode;
import com.example.csagent.chat.ModelServiceClient;
import com.example.csagent.chat.ModelServiceException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Java ↔ Python 契约（architecture.md §3）：请求字段、响应解析、SSE 帧、错误码映射。 */
class ModelServiceContractTest {

    /** 两侧共用的夹具真源：apps/contract/fixtures。 */
    private static final Path FIXTURES = Path.of("..", "contract", "fixtures");

    private final ObjectMapper mapper = new ObjectMapper();
    private final AtomicReference<String> planBody = new AtomicReference<>();
    private final AtomicReference<String> retrieveBody = new AtomicReference<>();
    private final AtomicReference<String> rerankBody = new AtomicReference<>();
    private final AtomicReference<String> generateBody = new AtomicReference<>();
    private final List<HttpServer> servers = new ArrayList<>();

    private String baseUrl;

    @AfterEach
    void stopServers() {
        servers.forEach(server -> server.stop(0));
        servers.clear();
    }

    private void startMainStub() throws IOException {
        HttpServer server = start();
        server.createContext(
                "/api/plan",
                exchange -> respondJson(exchange, 200, fixture("plan_response.json"), planBody));
        server.createContext(
                "/api/retrieve",
                exchange -> respondJson(exchange, 200, fixture("retrieve_response.json"), retrieveBody));
        server.createContext(
                "/api/rerank",
                exchange -> respondJson(exchange, 200, fixture("rerank_response.json"), rerankBody));
        server.createContext(
                "/api/generate",
                exchange -> respondText(
                        exchange, 200, "text/event-stream", fixture("generate_stream.txt"), generateBody));
        baseUrl = base(server);
    }

    private HttpServer start() throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        servers.add(server);
        server.start();
        return server;
    }

    private static String base(HttpServer server) {
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    private ModelServiceClient client() throws IOException {
        if (baseUrl == null) {
            startMainStub();
        }
        return new ModelServiceClient(baseUrl);
    }

    private static String fixture(String name) throws IOException {
        return Files.readString(FIXTURES.resolve(name), StandardCharsets.UTF_8);
    }

    private static List<String> fieldNames(JsonNode node) {
        List<String> names = new ArrayList<>();
        node.fieldNames().forEachRemaining(names::add);
        return names;
    }

    private JsonNode sent(AtomicReference<String> capture) throws IOException {
        return mapper.readTree(capture.get());
    }

    private static String readBody(HttpExchange exchange) throws IOException {
        try (var in = exchange.getRequestBody()) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static void respondJson(
            HttpExchange exchange, int status, String body, AtomicReference<String> capture)
            throws IOException {
        capture.set(readBody(exchange));
        respondText(exchange, status, "application/json", body, null);
    }

    private static void respondText(
            HttpExchange exchange, int status, String contentType, String body,
            AtomicReference<String> capture) throws IOException {
        if (capture != null) {
            capture.set(readBody(exchange));
        }
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", contentType);
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    /** 起一个只回答固定错误的 stub，用于错误码映射。 */
    private String stubReturning(String path, int status, String body) throws IOException {
        HttpServer server = start();
        server.createContext(path, exchange -> {
            readBody(exchange);
            respondText(exchange, status, "application/json", body, null);
        });
        return base(server);
    }

    @Test
    @DisplayName("plan sends query/session_id/history and parses the Python fixture")
    void plan_roundTrip() throws IOException {
        JsonNode plan = client().plan(
                "我的订单到哪了", "s-1",
                List.of(Map.of("role", "user", "content", "SO123456 什么时候到")));

        assertThat(plan.path("capabilities").size()).isEqualTo(2);
        assertThat(plan.path("topic").path("label").asText()).isEqualTo("物流");
        assertThat(plan.path("gate").path("order").path("allowed").asBoolean()).isTrue();
        assertThat(plan.path("source").asText()).isEqualTo("bert");

        JsonNode sent = sent(planBody);
        assertThat(fieldNames(sent)).containsExactly("query", "session_id", "history");
        assertThat(sent.path("history").get(0).path("role").asText()).isEqualTo("user");
    }

    @Test
    @DisplayName("blank session id is omitted from the plan payload")
    void plan_blankSession_omitsField() throws IOException {
        client().plan("你好", "  ", List.of());

        assertThat(fieldNames(sent(planBody))).containsExactly("query", "history");
    }

    @Test
    @DisplayName("CapabilityPlanner maps the fixture without dropping capabilities")
    void planner_mapsFixture() throws IOException {
        CapabilityPlanner planner = new CapabilityPlanner(client());

        CapabilityPlanner.Plan plan =
                planner.plan("我的订单到哪了", "s-1", List.of(ChatMessage.user("SO123456 什么时候到")));

        assertThat(plan.capabilities()).containsExactly("knowledge", "order");
        assertThat(plan.topic()).isEqualTo("物流");
        assertThat(plan.topicConfidence()).isEqualTo(0.7342);
        assertThat(plan.entities()).containsEntry("orderId", "SO123456");
        assertThat(plan.gate().get("order").allowed()).isTrue();
        assertThat(plan.source()).isEqualTo("bert");
        assertThat(plan.elapsedMs()).isEqualTo(12.4);
    }

    @Test
    @DisplayName("retrieve sends top_k/intent and parses candidates")
    void retrieve_roundTrip() throws IOException {
        JsonNode result = client().retrieve("怎么退货", 5, "售后");

        assertThat(result.path("candidates").get(0).path("text").asText())
                .startsWith("自签收之日起");
        assertThat(result.path("intent").asText()).isEqualTo("售后");

        JsonNode sent = sent(retrieveBody);
        assertThat(fieldNames(sent)).containsExactly("query", "top_k", "intent");
        assertThat(sent.path("top_k").asInt()).isEqualTo(5);
    }

    @Test
    @DisplayName("retrieve without intent omits the intent field")
    void retrieve_noIntent_omitsField() throws IOException {
        client().retrieve("怎么退货", 5, null);

        assertThat(fieldNames(sent(retrieveBody))).containsExactly("query", "top_k");
    }

    @Test
    @DisplayName("rerank sends candidates/top_k and parses selected")
    void rerank_roundTrip() throws IOException {
        JsonNode result = client().rerank("退款多久到账", List.of("a", "b"), 2);

        assertThat(result.path("selected").get(0).path("index").asInt()).isEqualTo(1);
        assertThat(result.path("selected").get(0).path("score").asDouble()).isEqualTo(0.9217);

        JsonNode sent = sent(rerankBody);
        assertThat(fieldNames(sent)).containsExactlyInAnyOrder("query", "candidates", "top_k");
        assertThat(sent.path("candidates").size()).isEqualTo(2);
    }

    @Test
    @DisplayName("generate forwards raw SSE lines through done")
    void generate_stream_roundTrip() throws IOException {
        try (Stream<String> lines = client().generateStream(
                List.of(Map.of("role", "user", "content", "订单到哪了")), 64)) {
            String wire = lines.collect(Collectors.joining("\n"));
            assertThat(wire)
                    .contains("data: {\"type\": \"token\"")
                    .contains("\"type\": \"done\"")
                    .contains("预计今天派送");
        }

        JsonNode sent = sent(generateBody);
        assertThat(fieldNames(sent)).containsExactlyInAnyOrder("messages", "max_new_tokens");
        assertThat(sent.path("max_new_tokens").asInt()).isEqualTo(64);
    }

    @Test
    @DisplayName("generate forwards an error frame verbatim")
    void generate_errorFrame_forwarded() throws IOException {
        HttpServer server = start();
        server.createContext("/api/generate", exchange -> {
            readBody(exchange);
            respondText(exchange, 200, "text/event-stream",
                    fixture("generate_error_stream.txt"), null);
        });

        try (Stream<String> lines = new ModelServiceClient(base(server)).generateStream(List.of(), 8)) {
            String wire = lines.collect(Collectors.joining("\n"));
            assertThat(wire)
                    .contains("\"type\": \"error\"")
                    .contains("\"partial\": \"半句\"");
        }
    }

    @Test
    @DisplayName("a 429 error envelope maps to UPSTREAM_RATE_LIMITED")
    void errorEnvelope_mapsCode() throws IOException {
        ModelServiceClient client =
                new ModelServiceClient(stubReturning("/api/plan", 429, fixture("error_envelope.json")));

        assertThatThrownBy(() -> client.plan("你好", null, List.of()))
                .isInstanceOfSatisfying(ModelServiceException.class,
                        ex -> assertThat(ex.code()).isEqualTo(ErrorCode.UPSTREAM_RATE_LIMITED));
    }

    @Test
    @DisplayName("an unknown error code falls back to INTERNAL")
    void unknownCode_fallsBackToInternal() throws IOException {
        ModelServiceClient client = new ModelServiceClient(stubReturning(
                "/api/plan", 500,
                "{\"code\":\"SOMETHING_NEW\",\"message\":\"x\",\"retriable\":false}"));

        assertThatThrownBy(() -> client.plan("你好", null, List.of()))
                .isInstanceOfSatisfying(ModelServiceException.class,
                        ex -> assertThat(ex.code()).isEqualTo(ErrorCode.INTERNAL));
    }

    @Test
    @DisplayName("a non-JSON error body falls back to INTERNAL")
    void nonJsonBody_fallsBackToInternal() throws IOException {
        ModelServiceClient client =
                new ModelServiceClient(stubReturning("/api/plan", 502, "bad gateway"));

        assertThatThrownBy(() -> client.plan("你好", null, List.of()))
                .isInstanceOfSatisfying(ModelServiceException.class,
                        ex -> assertThat(ex.code()).isEqualTo(ErrorCode.INTERNAL));
    }
}
