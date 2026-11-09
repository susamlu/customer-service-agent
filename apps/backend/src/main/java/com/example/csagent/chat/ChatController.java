package com.example.csagent.chat;

import com.example.csagent.obs.Trace;
import java.util.List;
import java.util.Map;
import org.slf4j.MDC;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

/** 会话入口：校验请求后交给编排器，SSE 事件由编排器产生。 */
@RestController
@RequestMapping("/api")
public class ChatController {

    /** 客服问题再啰嗦也该在一段话内说清；超长直接拒绝，不浪费一次模型调用。 */
    private static final int MAX_QUERY_CHARS = 512;

    private final ChatOrchestrator orchestrator;
    private final SessionStore store;
    private final ModelServiceClient client;

    public ChatController(
            ChatOrchestrator orchestrator, SessionStore store, ModelServiceClient client) {
        this.orchestrator = orchestrator;
        this.store = store;
        this.client = client;
    }

    /** SSE 流式回答：编排器串起模型原子能力与业务数据。 */
    @PostMapping(value = "/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public StreamingResponseBody chat(
            @RequestParam String sessionId,
            @RequestBody Map<String, String> body) {
        String query = body.getOrDefault("query", "").trim();
        String userId = body.getOrDefault("userId", "u_chen");
        // 流式 lambda 跑在异步线程，请求线程的 MDC 不会跟过去，先捕获再重新挂载。
        String traceId = Trace.current() == null ? Trace.newId() : Trace.current();
        return out -> {
            MDC.put(Trace.MDC_KEY, traceId);
            // SseChannel 负责心跳与串行写；close 会取消心跳任务。
            try (SseChannel channel = SseChannel.open(out)) {
                if (query.isEmpty()) {
                    error(channel, ErrorCode.INVALID_REQUEST);
                    return;
                }
                if (query.length() > MAX_QUERY_CHARS) {
                    error(channel, ErrorCode.REQUEST_TOO_LARGE);
                    return;
                }
                // 校验通过后才落库：空问题或超长问题不该污染会话历史。
                store.addUser(sessionId, query);
                orchestrator.handle(sessionId, query, userId, channel);
            } finally {
                MDC.remove(Trace.MDC_KEY);
            }
        };
    }

    @GetMapping("/history")
    public List<ChatMessage> history(@RequestParam String sessionId) {
        return store.history(sessionId);
    }

    @DeleteMapping("/history")
    public Map<String, String> clear(@RequestParam String sessionId) {
        store.clear(sessionId);
        return Map.of("status", "cleared");
    }

    @GetMapping(value = "/model-status", produces = MediaType.APPLICATION_JSON_VALUE)
    public String modelStatus() {
        return client.health();
    }

    /** 与 §3.5 信封一致，且与编排器产生的错误事件同形；traceId 让用户反馈能直达日志。 */
    private void error(SseChannel channel, ErrorCode code) {
        channel.send(ErrorEvents.payload(code));
    }
}
