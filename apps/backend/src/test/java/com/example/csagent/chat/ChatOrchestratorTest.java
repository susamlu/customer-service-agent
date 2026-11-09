package com.example.csagent.chat;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.example.csagent.mock.MockDataService;
import com.example.csagent.mock.model.OrderDetail;
import com.example.csagent.mock.model.UserContext;
import com.example.csagent.obs.ChatMetrics;
import com.example.csagent.ticket.Ticket;
import com.example.csagent.ticket.TicketStatus;
import com.example.csagent.ticket.TicketStore;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.core.io.ClassPathResource;

/**
 * 编排器分支（§5/§8）：路由出口、订单查不到的两种结局、可答性闸门三档。
 *
 * <pre>
 *  human      → 转人工固定话术，不调生成
 *  clarify    → 追问固定话术，不调生成
 *  order 查不到 + 无知识 → 固定话术收口（纯订单查询）
 *  order 查不到 + 有知识 → 事实注入上下文，照常生成
 *  verdict no_context   → 固定话术收口，不调生成
 *  verdict weak/answer  → 生成
 * </pre>
 */
class ChatOrchestratorTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final CapabilityPlanner planner = org.mockito.Mockito.mock(CapabilityPlanner.class);
    private final ModelServiceClient client = org.mockito.Mockito.mock(ModelServiceClient.class);
    private final MockDataService mockData = org.mockito.Mockito.mock(MockDataService.class);
    private final TicketStore ticketStore = org.mockito.Mockito.mock(TicketStore.class);
    private final SessionStore store = new SessionStore();
    private final ByteArrayOutputStream buffer = new ByteArrayOutputStream();

    private ChatOrchestrator orchestrator;

    @BeforeEach
    void setUp() {
        orchestrator = new ChatOrchestrator(
                client,
                planner,
                store,
                mockData,
                new FanOut(),
                new ContextAssembler(),
                new ChatMetrics(new SimpleMeterRegistry()),
                ticketStore,
                new ClassPathResource("prompts/chat-system.txt"));
        when(mockData.context(anyString())).thenReturn(Optional.empty());
    }

    @Test
    @DisplayName("human route opens a ticket and replies without calling generation")
    void humanRoute_transfersWithoutGeneration() throws IOException {
        when(planner.plan(anyString(), anyString(), anyList()))
                .thenReturn(plan(List.of("human"), Map.of()));
        when(ticketStore.create(anyString(), anyString(), anyString(), any()))
                .thenReturn(new Ticket(
                        42L, "s1", "u1", TicketStatus.PENDING, "我要转人工", null, 1L, 1L));

        run("我要转人工");

        assertThat(event("plan").path("route").asText()).isEqualTo("human");
        JsonNode ticket = event("ticket");
        assertThat(ticket.path("ticketId").asLong()).isEqualTo(42);
        assertThat(ticket.path("status").asText()).isEqualTo("PENDING");
        assertThat(event("reply").path("text").asText())
                .contains("转接人工")
                .contains("42");
        verify(client, never()).generateStream(any(), anyInt());
    }

    @Test
    @DisplayName("a repeated transfer reuses the open ticket instead of opening a second one")
    void repeatedTransfer_reusesOpenTicket() throws IOException {
        when(planner.plan(anyString(), anyString(), anyList()))
                .thenReturn(plan(List.of("human"), Map.of()));
        when(ticketStore.findOpen("s1")).thenReturn(Optional.of(new Ticket(
                7L, "s1", "u1", TicketStatus.IN_PROGRESS, "q", null, 1L, 1L)));

        run("再转一次人工");

        assertThat(event("ticket").path("ticketId").asLong()).isEqualTo(7);
        verify(ticketStore, never()).create(any(), any(), any(), any());
    }

    @Test
    @DisplayName("an empty capability set asks for clarification instead of guessing")
    void clarifyRoute_asksBack() throws IOException {
        when(planner.plan(anyString(), anyString(), anyList()))
                .thenReturn(plan(List.of(), Map.of()));

        run("嗯嗯");

        assertThat(event("plan").path("route").asText()).isEqualTo("clarify");
        JsonNode reply = event("reply");
        assertThat(reply.path("code").asText()).isEqualTo(ErrorCode.ROUTE_LOW_CONFIDENCE.name());
        verify(client, never()).generateStream(any(), anyInt());
    }

    @Test
    @DisplayName("a missing order with no knowledge is answered with the fixed notice")
    void missingOrderWithoutKnowledge_repliesDirectly() throws IOException {
        givenPlan(List.of("order"), Map.of("orderId", "SO999999"));
        when(mockData.context("u1")).thenReturn(Optional.of(userWith("SO123456")));

        run("订单 SO999999 到哪了");

        JsonNode reply = event("reply");
        assertThat(reply.path("text").asText()).contains("没有找到订单 SO999999");
        assertThat(reply.path("code").asText()).isEqualTo(ErrorCode.ORDER_NOT_FOUND.name());
        verify(client, never()).retrieve(anyString(), anyInt(), any(), any());
        verify(client, never()).generateStream(any(), anyInt());
    }

    @Test
    @DisplayName("a missing order with knowledge becomes a fact and still generates")
    void missingOrderWithKnowledge_injectsFactAndGenerates() throws IOException {
        givenPlan(List.of("order", "knowledge"), Map.of("orderId", "SO999999"));
        when(mockData.context("u1")).thenReturn(Optional.of(userWith("SO123456")));
        givenKnowledge(0.9, "退款 7 天内原路退回");
        givenGeneration("您好，这个订单号在系统里查不到。");

        run("订单 SO999999 能退款吗");

        JsonNode context = event("context");
        assertThat(context.path("sources").toString()).contains("knowledge");
        assertThat(context.path("notes").toString()).contains("订单号不存在");
        assertThat(event("verdict").path("verdict").asText()).isEqualTo("answer");
        assertThat(event("token").path("text").asText()).isEqualTo("您好");

        ArgumentCaptor<List<Map<String, String>>> messages = messagesCaptor();
        verify(client).generateStream(messages.capture(), anyInt());
        assertThat(userContent(messages.getValue()))
                .contains("[订单] 用户提供的订单号在系统中不存在")
                .contains("[知识] 退款 7 天内原路退回");
    }

    @Test
    @DisplayName("an account with no orders and no knowledge is answered with the fixed notice")
    void noOrderContextWithoutKnowledge_repliesDirectly() throws IOException {
        givenPlan(List.of("order"), Map.of());
        when(mockData.context("u1")).thenReturn(Optional.of(userWithOrders(List.of())));

        run("我的订单呢");

        JsonNode reply = event("reply");
        assertThat(reply.path("text").asText()).contains("暂时没有查到您的订单信息");
        assertThat(reply.path("code").asText()).isEqualTo(ErrorCode.ORDER_CONTEXT_MISSING.name());
        verify(client, never()).generateStream(any(), anyInt());
    }

    @Test
    @DisplayName("an empty recall ends at the no_context gate without generating")
    void emptyRecall_stopsAtGate() throws IOException {
        givenPlan(List.of("knowledge"), Map.of());
        when(client.retrieve(anyString(), anyInt(), any(), any()))
                .thenReturn(json("{\"candidates\":[],\"elapsed\":0.01}"));

        run("你们支持以旧换新吗");

        assertThat(event("verdict").path("verdict").asText()).isEqualTo("no_context");
        assertThat(event("verdict").path("reason").asText()).isEqualTo("empty");
        assertThat(event("reply").path("text").asText())
                .isEqualTo(AnswerabilityGate.NO_CONTEXT_REPLY);
        verify(client, never()).generateStream(any(), anyInt());
    }

    @Test
    @DisplayName("a weak recall is flagged but still generates")
    void weakRecall_generates() throws IOException {
        givenPlan(List.of("knowledge"), Map.of());
        givenKnowledge(0.60, "退款 7 天内原路退回");
        givenGeneration("退款一般 7 天内到账。");

        run("退款多久到账");

        assertThat(event("verdict").path("verdict").asText()).isEqualTo("weak");
        assertThat(event("verdict").path("reason").asText()).isEqualTo("recall_score");
        verify(client).generateStream(any(), anyInt());
    }

    @Test
    @DisplayName("a noop rerank shows the recall score instead of the placeholder zero")
    void noopRerank_backfillsRecallScore() throws IOException {
        // noop 精排的 score 恒为 0（表示这一步没有精排），面板不能用它当分数。
        givenPlan(List.of("knowledge"), Map.of());
        when(client.retrieve(anyString(), anyInt(), any(), any())).thenReturn(json(
                "{\"candidates\":[{\"text\":\"退款 7 天内原路退回\",\"intent\":\"knowledge\","
                        + "\"score\":0.77}],\"elapsed\":0.01}"));
        when(client.rerank(anyString(), anyList(), anyInt(), any())).thenReturn(json(
                "{\"selected\":[{\"text\":\"退款 7 天内原路退回\",\"score\":0.0}],"
                        + "\"scorer\":\"noop\",\"elapsed\":0.01}"));
        givenGeneration("7 天内到账。");

        run("退款多久到账");

        JsonNode rerank = event("rerank");
        assertThat(rerank.path("scorer").asText()).isEqualTo("noop");
        assertThat(rerank.path("selected").get(0).path("score").asDouble()).isEqualTo(0.77);
    }

    @Test
    @DisplayName("a cross-encoder rerank keeps its own score")
    void crossEncoderRerank_keepsRerankScore() throws IOException {
        givenPlan(List.of("knowledge"), Map.of());
        when(client.retrieve(anyString(), anyInt(), any(), any())).thenReturn(json(
                "{\"candidates\":[{\"text\":\"退款 7 天内原路退回\",\"intent\":\"knowledge\","
                        + "\"score\":0.77}],\"elapsed\":0.01}"));
        when(client.rerank(anyString(), anyList(), anyInt(), any())).thenReturn(json(
                "{\"selected\":[{\"text\":\"退款 7 天内原路退回\",\"score\":0.8}],"
                        + "\"scorer\":\"cross_encoder\",\"elapsed\":0.01}"));
        givenGeneration("7 天内到账。");

        run("退款多久到账");

        JsonNode rerank = event("rerank");
        assertThat(rerank.path("scorer").asText()).isEqualTo("cross_encoder");
        assertThat(rerank.path("selected").get(0).path("score").asDouble()).isEqualTo(0.8);
    }

    @Test
    @DisplayName("previous turns are replayed so a follow-up can resolve its reference")
    void previousTurnsAreReplayed() throws IOException {
        store.addUser("s1", "订单 SO123456 到哪了");
        store.addAssistant("s1", null, "已发货，预计明天送达");
        givenPlan(List.of("knowledge"), Map.of());
        givenKnowledge(0.9, "退款 7 天内原路退回");
        givenGeneration("一般 7 天内到账。");

        run("那退款呢");

        ArgumentCaptor<List<Map<String, String>>> messages = messagesCaptor();
        verify(client).generateStream(messages.capture(), anyInt());
        List<Map<String, String>> value = messages.getValue();
        assertThat(value).extracting(message -> message.get("role"))
                .containsExactly("system", "user", "assistant", "user");
        assertThat(value.get(1).get("content")).contains("SO123456");
        assertThat(value.get(3).get("content"))
                .contains("早前对话仅作背景")
                .contains("用户问题：那退款呢");
    }

    @Test
    @DisplayName("the current question is not replayed as history")
    void currentQuestionIsNotDuplicated() throws IOException {
        // 控制器在调用编排器前就把本轮问题落了库；生成时不能把它当历史再喂一遍。
        store.addUser("s1", "上一句问题");
        store.addUser("s1", "那退款呢");
        givenPlan(List.of("knowledge"), Map.of());
        givenKnowledge(0.9, "退款 7 天内原路退回");
        givenGeneration("7 天内。");

        run("那退款呢");

        ArgumentCaptor<List<Map<String, String>>> messages = messagesCaptor();
        verify(client).generateStream(messages.capture(), anyInt());
        long occurrences = messages.getValue().stream()
                .filter(message -> "user".equals(message.get("role")))
                .filter(message -> message.get("content").contains("那退款呢"))
                .count();
        assertThat(occurrences).isEqualTo(1);
    }

    @Test
    @DisplayName("replayed history is capped and each entry is truncated")
    void replayedHistoryIsBounded() throws IOException {
        for (int i = 0; i < 10; i++) {
            store.addUser("s1", "msg-" + i);
        }
        givenPlan(List.of("knowledge"), Map.of());
        givenKnowledge(0.9, "退款 7 天内原路退回");
        givenGeneration("7 天内。");

        run("新问题");

        ArgumentCaptor<List<Map<String, String>>> messages = messagesCaptor();
        verify(client).generateStream(messages.capture(), anyInt());
        List<Map<String, String>> value = messages.getValue();
        // system + 最近 6 条历史 + 本轮 user
        assertThat(value).hasSize(8);
        assertThat(value.get(1).get("content")).isEqualTo("msg-4");
    }

    // ---------- helpers ----------

    private void run(String query) throws IOException {
        try (SseChannel channel = SseChannel.open(buffer)) {
            orchestrator.handle("s1", query, "u1", channel);
        }
    }

    private void givenPlan(List<String> capabilities, Map<String, String> entities) {
        when(planner.plan(anyString(), anyString(), anyList()))
                .thenReturn(plan(capabilities, entities));
    }

    private void givenKnowledge(double score, String text) throws IOException {
        when(client.retrieve(anyString(), anyInt(), any(), any())).thenReturn(json(
                "{\"candidates\":[{\"text\":\"" + text + "\",\"intent\":\"knowledge\","
                        + "\"score\":" + score + "}],\"elapsed\":0.01}"));
        when(client.rerank(anyString(), anyList(), anyInt(), any())).thenReturn(json(
                "{\"selected\":[{\"text\":\"" + text + "\",\"score\":0.8}],"
                        + "\"scorer\":\"cross_encoder\",\"elapsed\":0.01}"));
    }

    private void givenGeneration(String reply) {
        when(client.generateStream(any(), anyInt())).thenReturn(Stream.of(
                "data: {\"type\":\"token\",\"text\":\"您好\"}", "",
                "data: {\"type\":\"done\",\"reply\":\"" + reply + "\"}", ""));
    }

    private static CapabilityPlanner.Plan plan(
            List<String> capabilities, Map<String, String> entities) {
        return new CapabilityPlanner.Plan(
                List.copyOf(capabilities),
                Map.of("knowledge", 0.9, "order", 0.9, "human", 0.1),
                "退款",
                0.9,
                Map.copyOf(entities),
                Map.of(),
                "bert",
                20.0);
    }

    private static UserContext userWith(String orderId) {
        return userWithOrders(List.of(order(orderId)));
    }

    private static UserContext userWithOrders(List<OrderDetail> orders) {
        return new UserContext("u1", "张三", "小张", "gold", "2024-01-01", orders);
    }

    private static OrderDetail order(String orderId) {
        return new OrderDetail(
                orderId, "SHIPPED", "已发货", "2026-11-01", 7, 199.0, "支付宝",
                3, false, null, "",
                List.of(new OrderDetail.ItemDetail(
                        "P1", "卫衣", "服装", 199.0, 1, true, 15, true, "吊牌完好")),
                new OrderDetail.LogisticsDetail(
                        "顺丰", "SF123",
                        List.of(new OrderDetail.LogisticsDetail.TraceDetail(
                                "2026-11-05", 3, "杭州", "已揽收"))));
    }

    private static JsonNode json(String raw) throws IOException {
        return MAPPER.readTree(raw);
    }

    private JsonNode event(String type) throws IOException {
        List<JsonNode> events = new ArrayList<>();
        for (String frame : buffer.toString(StandardCharsets.UTF_8).split("\n\n")) {
            String trimmed = frame.strip();
            if (trimmed.startsWith("data: ")) {
                events.add(MAPPER.readTree(trimmed.substring(6)));
            }
        }
        return events.stream()
                .filter(node -> type.equals(node.path("type").asText()))
                .findFirst()
                .orElseThrow(() -> new AssertionError("no " + type + " event in " + buffer));
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static ArgumentCaptor<List<Map<String, String>>> messagesCaptor() {
        return (ArgumentCaptor) ArgumentCaptor.forClass(List.class);
    }

    private static String userContent(List<Map<String, String>> messages) {
        return messages.stream()
                .filter(message -> "user".equals(message.get("role")))
                .map(message -> message.get("content"))
                .findFirst()
                .orElseThrow();
    }
}
