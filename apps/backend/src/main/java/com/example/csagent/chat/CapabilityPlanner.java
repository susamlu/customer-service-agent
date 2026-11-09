package com.example.csagent.chat;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Component;

/**
 * 能力规划：把 /api/plan 的原始响应收敛成网关可执行的规划结果（architecture.md §3.1/§4.4）。
 *
 * <pre>
 *  query + 会话历史
 *        │
 *        ▼
 *  POST /api/plan ──► capabilities / topic / entities / gate
 *        │
 *        ▼
 *  Plan（不可变值对象）──► 编排器按能力集调度
 * </pre>
 *
 * <p>门控在模型服务侧完成（单一事实来源，避免两处规则漂移），这里只做契约校验：
 * 未知能力直接丢弃；全部丢弃时降级为 clarify。
 */
@Component
public class CapabilityPlanner {

    private static final Set<String> KNOWN_CAPABILITIES = Set.of("knowledge", "order", "human");
    /** 只把最近若干轮历史送去做门控：订单号回溯只关心近距离上下文。 */
    private static final int HISTORY_LIMIT = 20;

    private final ModelServiceClient client;

    public CapabilityPlanner(ModelServiceClient client) {
        this.client = client;
    }

    public Plan plan(String query, String sessionId, List<ChatMessage> history) {
        JsonNode raw = client.plan(query, sessionId, recentHistory(history));
        return parse(raw);
    }

    private Plan parse(JsonNode raw) {
        List<String> capabilities = new ArrayList<>();
        for (JsonNode capability : raw.path("capabilities")) {
            String name = capability.asText("");
            if (KNOWN_CAPABILITIES.contains(name) && !capabilities.contains(name)) {
                capabilities.add(name);
            }
        }

        Map<String, Double> routeScores = new LinkedHashMap<>();
        JsonNode scores = raw.path("route_scores");
        if (scores.isObject()) {
            scores.fields().forEachRemaining(entry ->
                    routeScores.put(entry.getKey(), entry.getValue().asDouble(0)));
        }

        Map<String, String> entities = new LinkedHashMap<>();
        JsonNode entityNode = raw.path("entities");
        if (entityNode.isObject()) {
            entityNode.fields().forEachRemaining(entry ->
                    entities.put(entry.getKey(), entry.getValue().asText("")));
        }

        Map<String, Plan.Gate> gate = new LinkedHashMap<>();
        JsonNode gateNode = raw.path("gate");
        if (gateNode.isObject()) {
            gateNode.fields().forEachRemaining(entry -> gate.put(
                    entry.getKey(),
                    new Plan.Gate(
                            entry.getValue().path("allowed").asBoolean(false),
                            entry.getValue().path("reason").asText(""))));
        }

        JsonNode topic = raw.path("topic");
        String topicLabel = topic.path("label").asText("");

        return new Plan(
                List.copyOf(capabilities),
                Map.copyOf(routeScores),
                topicLabel.isBlank() ? null : topicLabel,
                topic.path("confidence").asDouble(0),
                Map.copyOf(entities),
                Map.copyOf(gate),
                raw.path("source").asText(""),
                raw.path("elapsed_ms").asDouble(0));
    }

    /** 会话消息转成模型服务约定的 {role, content}；只取最近 HISTORY_LIMIT 条。 */
    private List<Map<String, String>> recentHistory(List<ChatMessage> history) {
        if (history == null || history.isEmpty()) {
            return List.of();
        }
        List<ChatMessage> recent = history.subList(
                Math.max(0, history.size() - HISTORY_LIMIT), history.size());
        List<Map<String, String>> messages = new ArrayList<>(recent.size());
        for (ChatMessage message : recent) {
            messages.add(Map.of("role", message.role(), "content", message.text()));
        }
        return messages;
    }

    /** /api/plan 的收敛结果；capabilities 为空表示无法路由，走澄清。 */
    public record Plan(
            List<String> capabilities,
            Map<String, Double> routeScores,
            String topic,
            double topicConfidence,
            Map<String, String> entities,
            Map<String, Gate> gate,
            String source,
            double elapsedMs) {

        public record Gate(boolean allowed, String reason) {}

        /**
         * 单一标签，仅供三类用途：human/clarify 的分支判断、plan 事件里的 route 元数据、
         * 会话记录。真正的调度走 capabilities 并行 fan-out（ChatOrchestrator.handleAnswer），
         * 不要把本方法当成执行入口。
         *
         * 优先级 human → order → knowledge → clarify 只是「这个会话看起来属于谁」的展示口径。
         */
        public String primaryRoute() {
            if (capabilities.contains("human")) {
                return "human";
            }
            if (capabilities.contains("order")) {
                return "order";
            }
            if (capabilities.contains("knowledge")) {
                return "knowledge";
            }
            return "clarify";
        }
    }
}
