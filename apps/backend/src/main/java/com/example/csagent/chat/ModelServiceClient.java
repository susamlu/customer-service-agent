package com.example.csagent.chat;

import com.example.csagent.obs.Trace;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/** 调用 Python 模型服务的原子能力：意图识别、召回、重排、流式生成。 */
@Component
public class ModelServiceClient {

    private static final Duration JSON_TIMEOUT = Duration.ofSeconds(120);
    // 规划是整条链路的第一跳，服务端实测 15–20ms；卡住时必须尽快失败，
    // 不能让用户盯着 120s 的默认超时等一个本该瞬间返回的结果。
    private static final Duration PLAN_TIMEOUT = Duration.ofSeconds(3);
    private static final Duration STREAM_TIMEOUT = Duration.ofSeconds(300);

    private final HttpClient http = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(5))
            .build();
    private final ObjectMapper mapper = new ObjectMapper();
    private final String baseUrl;

    public ModelServiceClient(@Value("${model.service.base-url}") String baseUrl) {
        this.baseUrl = baseUrl;
    }

    /** 一次调用拿到能力集、知识主题、业务实体与门控结果（architecture.md §3.1）。 */
    public JsonNode plan(String query, String sessionId, List<Map<String, String>> history) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("query", query);
        if (sessionId != null && !sessionId.isBlank()) {
            payload.put("session_id", sessionId);
        }
        payload.put("history", history);
        return postJson("/api/plan", payload, PLAN_TIMEOUT);
    }

    /** intent 非空时，模型服务对同意图分区的候选加权。 */
    public JsonNode retrieve(String query, int topK, String intent) {
        return retrieve(query, topK, intent, JSON_TIMEOUT);
    }

    /** 带独立超时的召回：fan-out 里每个能力有自己的预算（§5.2）。 */
    public JsonNode retrieve(String query, int topK, String intent, Duration timeout) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("query", query);
        payload.put("top_k", topK);
        if (intent != null && !intent.isBlank()) {
            payload.put("intent", intent);
        }
        return postJson("/api/retrieve", payload, timeout);
    }

    public JsonNode rerank(String query, List<String> candidates, int topK) {
        return rerank(query, candidates, topK, JSON_TIMEOUT);
    }

    /** 带独立超时的重排：超时由调用方降级为召回顺序（§5.2）。 */
    public JsonNode rerank(String query, List<String> candidates, int topK, Duration timeout) {
        return postJson(
                "/api/rerank",
                Map.of("query", query, "candidates", candidates, "top_k", topK),
                timeout);
    }

    /** 生成 SSE 行流。调用方负责关闭。 */
    public Stream<String> generateStream(List<Map<String, String>> messages, int maxNewTokens) {
        String body;
        try {
            body = mapper.writeValueAsString(
                    Map.of("messages", messages, "max_new_tokens", maxNewTokens));
        } catch (IOException e) {
            throw new IllegalStateException("生成请求序列化失败", e);
        }
        return postStream("/api/generate", body);
    }

    public String health() {
        try {
            HttpRequest request = requestTo("/healthz")
                    .timeout(Duration.ofSeconds(3))
                    .GET()
                    .build();
            return http.send(request, HttpResponse.BodyHandlers.ofString()).body();
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            return "{\"status\":\"down\"}";
        }
    }

    private JsonNode postJson(String path, Object payload) {
        return postJson(path, payload, JSON_TIMEOUT);
    }

    private JsonNode postJson(String path, Object payload, Duration timeout) {
        try {
            String body = mapper.writeValueAsString(payload);
            HttpRequest request = requestTo(path)
                    .header("Content-Type", "application/json")
                    .timeout(timeout)
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> response =
                    http.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw decodeError(response.statusCode(), response.body());
            }
            return mapper.readTree(response.body());
        } catch (HttpTimeoutException e) {
            throw new ModelServiceException(ErrorCode.UPSTREAM_TIMEOUT, "连接模型服务超时", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ModelServiceException(ErrorCode.INTERNAL, "调用模型服务被中断", e);
        } catch (IOException e) {
            throw new ModelServiceException(
                    ErrorCode.INTERNAL, "模型服务不可用，请确认 8000 端口服务已启动", e);
        }
    }

    private Stream<String> postStream(String path, String body) {
        try {
            HttpRequest request = requestTo(path)
                    .header("Content-Type", "application/json")
                    .timeout(STREAM_TIMEOUT)
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<Stream<String>> response =
                    http.send(request, HttpResponse.BodyHandlers.ofLines());
            if (response.statusCode() != 200) {
                String errorBody = response.body().collect(Collectors.joining());
                throw decodeError(response.statusCode(), errorBody);
            }
            return response.body();
        } catch (HttpTimeoutException e) {
            throw new ModelServiceException(ErrorCode.UPSTREAM_TIMEOUT, "连接模型服务超时", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ModelServiceException(ErrorCode.INTERNAL, "调用模型服务被中断", e);
        } catch (IOException e) {
            throw new ModelServiceException(
                    ErrorCode.INTERNAL, "模型服务不可用，请确认 8000 端口服务已启动", e);
        }
    }

    /**
     * 统一构造请求并带上 trace_id（§11）。上游据此把日志挂到同一条链路上；
     * 没有请求上下文时（如启动自检）不带这个头，也不能因此失败。
     */
    private HttpRequest.Builder requestTo(String path) {
        HttpRequest.Builder builder = HttpRequest.newBuilder().uri(URI.create(baseUrl + path));
        String traceId = Trace.current();
        return traceId == null ? builder : builder.header(Trace.HEADER, traceId);
    }

    /** 解析 §3.5 的错误信封；解析失败也不能把裸状态码当话术给用户。 */
    private ModelServiceException decodeError(int status, String body) {
        try {
            JsonNode envelope = mapper.readTree(body);
            ErrorCode code = ErrorCode.fromCode(envelope.path("code").asText(null));
            String detail = envelope.path("message").asText("");
            return new ModelServiceException(
                    code, "模型服务 " + status + "：" + detail, null);
        } catch (IOException e) {
            return new ModelServiceException(ErrorCode.INTERNAL, "模型服务 " + status, e);
        }
    }
}
