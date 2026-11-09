package com.example.csagent.chat;

import com.example.csagent.mock.MockDataService;
import com.example.csagent.mock.model.OrderDetail;
import com.example.csagent.mock.model.UserContext;
import com.example.csagent.obs.ChatMetrics;
import com.example.csagent.obs.Trace;
import com.example.csagent.ticket.Ticket;
import com.example.csagent.ticket.TicketStore;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.UncheckedIOException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

/**
 * 业务编排：能力规划 → 并行取数/检索 → 上下文拼装 → 生成（architecture.md §5）。
 *
 * <pre>
 *  guard → history → plan → emit plan
 *                          │
 *          ┌───────────────┴───────────────┐
 *          ▼                               ▼
 *   order.lookup (300ms)            retrieve → rerank (500+500ms)
 *          └───────────────┬───────────────┘
 *                          ▼
 *                  ContextAssembler（去重/裁剪/来源/冲突）
 *                          ▼
 *                  generate（SSE token 流）
 * </pre>
 *
 * <p>能力之间没有先后依赖，全部并发跑；每个能力独立超时，慢的那个只丢自己。
 * 订单匹配、上下文拼装、事件流组织都在网关完成，模型服务只提供原子推理能力。
 */
@Component
public class ChatOrchestrator {

    private static final Logger log = LoggerFactory.getLogger(ChatOrchestrator.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    // 只认 SO 开头的订单号。裸 11 位数字会误吞手机号，客服场景里代价太高。
    private static final Pattern ORDER_ID =
            Pattern.compile("SO\\d{6,}", Pattern.CASE_INSENSITIVE);
    private static final int RETRIEVE_TOP_K = 10;
    private static final int RERANK_TOP_K = 2;
    private static final int BRIEF_LIMIT = 90;
    // 客服回答用 128 token（约 150 字）封顶，配合 prompt 里的字数约束，
    // 让模型自然收口而不是被硬截断。
    private static final int MAX_NEW_TOKENS = 128;
    // 每个能力各自的预算（§5.2）。真实订单 API 可放宽到 800ms，mock 阶段 300ms 足够。
    private static final Duration ORDER_TIMEOUT = Duration.ofMillis(300);
    private static final Duration RETRIEVE_TIMEOUT = Duration.ofMillis(500);
    private static final Duration RERANK_TIMEOUT = Duration.ofMillis(500);
    // 工单上下文快照：带最近这段会话，人工接手不用重放整段。
    private static final int TICKET_HISTORY_TURNS = 10;
    private static final int TICKET_CONTEXT_LIMIT = 2000;
    // 生成时的短期记忆：最近 6 条消息（3 轮），每条截断，给"它到哪了"这类指代留落点。
    private static final int MEMORY_MESSAGES = 6;
    private static final int MEMORY_CHARS = 120;

    private final ModelServiceClient client;
    private final CapabilityPlanner planner;
    private final SessionStore store;
    private final MockDataService mockData;
    private final FanOut fanOut;
    private final ContextAssembler assembler;
    private final ChatMetrics metrics;
    private final TicketStore ticketStore;
    private final String chatSystem;

    public ChatOrchestrator(
            ModelServiceClient client,
            CapabilityPlanner planner,
            SessionStore store,
            MockDataService mockData,
            FanOut fanOut,
            ContextAssembler assembler,
            ChatMetrics metrics,
            TicketStore ticketStore,
            @Value("classpath:prompts/chat-system.txt") Resource chatPrompt) {
        this.client = client;
        this.planner = planner;
        this.store = store;
        this.mockData = mockData;
        this.fanOut = fanOut;
        this.assembler = assembler;
        this.metrics = metrics;
        this.ticketStore = ticketStore;
        this.chatSystem = read(chatPrompt);
    }

    public void handle(String sessionId, String query, String userId, SseChannel out) {
        // 本轮的唯一标识：意图归档按它索引，同一会话并发两轮才不会互相覆盖。
        // 优先复用 trace_id，让归档能与日志对上；直接调用（单测）时没有 MDC，退化为 UUID。
        String turnId = Trace.current() == null ? UUID.randomUUID().toString() : Trace.current();
        try {
            out.send(Map.of("type", "stage", "name", "intent", "status", "start"));
            long planStartedAt = System.nanoTime();
            CapabilityPlanner.Plan plan = planner.plan(query, sessionId, store.history(sessionId));
            String route = plan.primaryRoute();
            // 网关侧实测耗时（含网络），与 SSE 里的 elapsed（服务端推理）分开看。
            metrics.plan(Duration.ofNanos(System.nanoTime() - planStartedAt), plan.source(), route);
            out.send(planEvent(plan, route));
            store.recordIntent(turnId, route, plan.topic());

            switch (route) {
                case "human" -> handleTransfer(sessionId, turnId, query, userId, out);
                case "clarify" -> handleClarify(sessionId, turnId, out);
                // order / knowledge 可能同时命中：不选边，能力集里的全部并发执行。
                default -> handleAnswer(sessionId, turnId, query, userId, plan, out);
            }
            metrics.request("ok");
        } catch (UncheckedIOException e) {
            // 客户端主动断开：流已经不可写，任何兜底事件都会再次抛错。
            metrics.request("disconnect");
            log.debug("client disconnected session={}", sessionId);
        } catch (ModelServiceException e) {
            metrics.request("error");
            log.error("chat failed code={} session={}", e.code(), sessionId, e);
            writeError(out, e.code());
        } catch (Exception e) {
            // 铁律：异常的 getMessage() 绝不进响应体（§9）。
            metrics.request("error");
            log.error("chat failed session={}", sessionId, e);
            writeError(out, ErrorCode.INTERNAL);
        }
    }

    /**
     * 订单 / 知识能力并发执行，再合并上下文生成（§5.1）。
     *
     * <p>订单的两种"查不到"分两条路：知识没参与时用固定话术提前收口（没有可合并的
     * 内容，模型只会复述）；知识参与时把"订单号不存在 / 账号无订单"作为事实注入
     * 上下文，让知识结果照常合并，也绝不给模型自由发挥编造物流的空间（§8 铁律）。
     * 超时/异常降级为"暂时不可用"，同样不阻断知识能力。
     */
    private void handleAnswer(
            String sessionId,
            String turnId,
            String query,
            String userId,
            CapabilityPlanner.Plan plan,
            SseChannel out)
            throws IOException {
        UserContext ctx = mockData.context(userId).orElse(null);
        String orderId = resolveOrderId(query, plan.entities().get("orderId"));

        List<FanOut.Task> tasks = new ArrayList<>(2);
        if (plan.capabilities().contains("order")) {
            tasks.add(new FanOut.Task(
                    "order", ORDER_TIMEOUT, () -> lookupOrder(ctx, orderId)));
        }
        if (plan.capabilities().contains("knowledge")) {
            // 召回与重排各自 500ms（§5.2）：外层预算取两级之和，两级各自超时
            // 由 ModelServiceClient 的 per-request timeout 保证。
            tasks.add(new FanOut.Task(
                    "knowledge",
                    RETRIEVE_TIMEOUT.plus(RERANK_TIMEOUT),
                    () -> retrieveAndRerank(query, plan.topic())));
        }
        Map<String, FanOut.Outcome> results = fanOut.byName(fanOut.all(tasks));

        FanOut.Outcome orderOutcome = results.get("order");
        OrderLookup lookup = orderOutcome != null && orderOutcome.ok()
                ? orderOutcome.valueAs(OrderLookup.class)
                : null;
        if (orderOutcome != null) {
            metrics.capability(seconds(orderOutcome.elapsedSec()), "order", orderTag(orderOutcome, lookup));
        }
        // 纯订单查询（知识没计划）保留固定话术：句子里没有可合并的知识，
        // 交给模型只会把同一件事说得更绕。知识也参与时，"查不到"是一个事实，
        // 必须注入上下文，否则模型会拿常识编造物流（§8 铁律）。
        boolean knowledgePlanned = plan.capabilities().contains("knowledge");
        OrderDetail order = lookup instanceof OrderLookup.Found found ? found.order() : null;
        ContextAssembler.OrderState orderState = ContextAssembler.OrderState.NONE;
        if (orderOutcome != null && !orderOutcome.ok()) {
            orderState = ContextAssembler.OrderState.UNAVAILABLE;
            log.warn(
                    "order capability degraded session={} timedOut={} error={}",
                    sessionId,
                    orderOutcome.timedOut(),
                    orderOutcome.error() == null ? "" : orderOutcome.error().toString());
        } else if (order != null) {
            writeEvent(out, Map.of(
                    "type", "order",
                    "result", orderSummary(order),
                    "elapsed", orderOutcome.elapsedSec()));
        } else if (lookup instanceof OrderLookup.Missing missing) {
            if (!knowledgePlanned) {
                reply(sessionId, turnId, out,
                        "没有找到订单 " + missing.orderId() + "，请核对订单号后重试，"
                                + "或回复\"转人工\"由客服协助查询。",
                        ErrorCode.ORDER_NOT_FOUND);
                return;
            }
            orderState = ContextAssembler.OrderState.MISSING;
            log.info("order missing session={} orderId={}", sessionId, missing.orderId());
        } else if (lookup instanceof OrderLookup.NoContext) {
            if (!knowledgePlanned) {
                reply(sessionId, turnId, out,
                        "暂时没有查到您的订单信息，请确认账号后重试，或需要我为您转接人工客服。",
                        ErrorCode.ORDER_CONTEXT_MISSING);
                return;
            }
            orderState = ContextAssembler.OrderState.NO_CONTEXT;
        }

        FanOut.Outcome knowledgeOutcome = results.get("knowledge");
        Knowledge knowledge = knowledgeOutcome != null && knowledgeOutcome.ok()
                ? knowledgeOutcome.valueAs(Knowledge.class)
                : null;
        if (knowledgeOutcome != null && !knowledgeOutcome.ok()) {
            // 走到这里的失败只可能来自召回：重排异常已在能力链内部降级。
            metrics.capability(seconds(knowledgeOutcome.elapsedSec()), "retrieve", "degraded");
            log.warn(
                    "retrieval degraded session={} timedOut={} error={}",
                    sessionId,
                    knowledgeOutcome.timedOut(),
                    knowledgeOutcome.error() == null ? "" : knowledgeOutcome.error().toString());
        }
        if (knowledge != null) {
            writeEvent(out, Map.of(
                    "type", "retrieve",
                    "candidates", knowledge.candidates(),
                    "elapsed", knowledge.retrieveElapsed()));
            writeEvent(out, Map.of(
                    "type", "rerank",
                    "selected", knowledge.selected(),
                    "elapsed", knowledge.rerankElapsed()));
            if (knowledge.selectedTexts().isEmpty()) {
                // 非故障（200，不可重试）：用常识回答即可，不弹错误气泡。
                log.warn("RETRIEVAL_EMPTY session={} query={}", sessionId, query);
            }
        }

        ContextAssembler.Assembled assembled = assembler.assemble(
                order,
                orderState,
                knowledge == null ? List.of() : knowledge.selectedTexts());
        writeEvent(out, contextEvent(assembled));

        // 可答性闸门（§5.4）：既没可信知识、也没订单事实时不调生成，
        // 用固定话术收口，避免模型拿常识编造。"查不到订单"本身也是事实，
        // 模型据此能给出确定回答，所以一样算 orderFound。
        boolean orderFact = order != null
                || orderState == ContextAssembler.OrderState.MISSING
                || orderState == ContextAssembler.OrderState.NO_CONTEXT;
        AnswerabilityGate.Decision verdict = AnswerabilityGate.decide(
                knowledgePlanned,
                knowledge == null && knowledgeOutcome != null,
                orderFact,
                maxRecallScore(knowledge));
        metrics.verdict(
                verdict.verdict().name().toLowerCase(Locale.ROOT),
                verdict.reason().name().toLowerCase(Locale.ROOT));
        writeEvent(out, verdictEvent(verdict));
        if (verdict.verdict() == AnswerabilityGate.Verdict.NO_CONTEXT) {
            log.info(
                    "verdict no_context session={} reason={} score={}",
                    sessionId,
                    verdict.reason(),
                    verdict.score());
            reply(sessionId, turnId, out, AnswerabilityGate.NO_CONTEXT_REPLY,
                    ErrorCode.NO_GROUNDED_CONTEXT);
            return;
        }

        writeEvent(out, Map.of("type", "stage", "name", "generate", "status", "start"));
        streamGenerate(sessionId, turnId, out, chatMessages(sessionId, assembled.text(), query));
    }

    /**
     * 取最近若干条会话作为多轮上下文。条数和单条长度都封顶：记忆是辅助，
     * 不能把 token 预算和历史噪声一起塞给模型。
     */
    private List<Map<String, String>> recentDialogue(String sessionId, String query) {
        List<ChatMessage> history = store.history(sessionId);
        int end = history.size();
        if (end > 0) {
            ChatMessage last = history.get(end - 1);
            if ("user".equals(last.role()) && last.text().equals(query)) {
                end--;
            }
        }
        int start = Math.max(0, end - MEMORY_MESSAGES);
        return history.subList(start, end).stream()
                .map(message -> Map.of(
                        "role", message.role(),
                        "content", truncate(message.text(), MEMORY_CHARS)))
                .collect(Collectors.toList());
    }

    /** 订单号缺失时取最近一笔；显式订单号查不到如实返回 Missing，绝不回退。 */
    private OrderLookup lookupOrder(UserContext ctx, String orderId) {
        if (ctx == null || ctx.orders().isEmpty()) {
            return new OrderLookup.NoContext();
        }
        if (orderId == null) {
            return new OrderLookup.Found(mostRecentOrder(ctx.orders()));
        }
        OrderDetail order = findOrder(ctx.orders(), orderId);
        return order == null ? new OrderLookup.Missing(orderId) : new OrderLookup.Found(order);
    }

    /** 召回 → 重排组成一条能力链；重排失败/超时降级为召回顺序，不阻断生成（§5.2）。 */
    private Knowledge retrieveAndRerank(String query, String topic) {
        long retrieveStartedAt = System.nanoTime();
        JsonNode retrieved = client.retrieve(query, RETRIEVE_TOP_K, topic, RETRIEVE_TIMEOUT);
        List<String> candidateTexts = new ArrayList<>();
        List<Map<String, Object>> candidates = new ArrayList<>();
        for (JsonNode candidate : retrieved.path("candidates")) {
            String text = candidate.path("text").asText("");
            candidateTexts.add(text);
            candidates.add(Map.<String, Object>of(
                    "text", brief(text),
                    "intent", candidate.path("intent").asText(""),
                    "score", candidate.path("score").asDouble(0)));
        }
        double retrieveElapsed = retrieved.path("elapsed").asDouble(0);
        metrics.capability(
                Duration.ofNanos(System.nanoTime() - retrieveStartedAt),
                "retrieve",
                candidateTexts.isEmpty() ? "empty" : "ok");
        if (candidateTexts.isEmpty()) {
            return new Knowledge(candidates, List.of(), List.of(), retrieveElapsed, 0);
        }

        long rerankStartedAt = System.nanoTime();
        JsonNode reranked;
        try {
            reranked = client.rerank(query, candidateTexts, RERANK_TOP_K, RERANK_TIMEOUT);
        } catch (ModelServiceException e) {
            metrics.capability(
                    Duration.ofNanos(System.nanoTime() - rerankStartedAt), "rerank", "degraded");
            log.warn("rerank degraded, fall back to recall order: {}", e.getMessage());
            return new Knowledge(
                    candidates,
                    candidates.stream().limit(RERANK_TOP_K).toList(),
                    candidateTexts.stream().limit(RERANK_TOP_K).toList(),
                    retrieveElapsed,
                    0);
        }
        metrics.capability(Duration.ofNanos(System.nanoTime() - rerankStartedAt), "rerank", "ok");

        List<String> selectedTexts = new ArrayList<>();
        List<Map<String, Object>> selected = new ArrayList<>();
        for (JsonNode node : reranked.path("selected")) {
            String text = node.path("text").asText("");
            selectedTexts.add(text);
            selected.add(Map.<String, Object>of(
                    "text", brief(text), "score", node.path("score").asDouble(0)));
        }
        return new Knowledge(
                candidates, selected, selectedTexts, retrieveElapsed,
                reranked.path("elapsed").asDouble(0));
    }

    /** 一条能力链的产出：候选与精选各自保留"事件形态"和"原文"，前者给 UI，后者给生成。 */
    private record Knowledge(
            List<Map<String, Object>> candidates,
            List<Map<String, Object>> selected,
            List<String> selectedTexts,
            double retrieveElapsed,
            double rerankElapsed) {}

    /**
     * 转人工：落一张工单再回执（§6）。
     *
     * <p>同一会话已有未结束工单时复用而不是新建——用户连点两次「转人工」不该在客服
     * 后台变成两张待办。上下文快照取最近的会话，人工接手前就能看到前因后果。
     */
    private void handleTransfer(
            String sessionId, String turnId, String query, String userId, SseChannel out) {
        Ticket ticket = ticketStore.findOpen(sessionId)
                .orElseGet(() -> ticketStore.create(
                        sessionId, userId, query, ticketContext(sessionId)));
        out.send(Map.of(
                "type", "ticket",
                "ticketId", ticket.id(),
                "status", ticket.status().name()));
        reply(sessionId, turnId, out, "已为您转接人工客服，工单号 " + ticket.id() + "，请稍候。");
    }

    /** 工单上下文：最近若干条会话，截断到 DB 列宽以内。 */
    private String ticketContext(String sessionId) {
        List<ChatMessage> history = store.history(sessionId);
        if (history.isEmpty()) {
            return null;
        }
        String joined = history.stream()
                .skip(Math.max(0, history.size() - TICKET_HISTORY_TURNS))
                .map(message -> message.role() + ": " + message.text())
                .collect(Collectors.joining("\n"));
        return joined.length() <= TICKET_CONTEXT_LIMIT
                ? joined
                : joined.substring(0, TICKET_CONTEXT_LIMIT);
    }

    /** 能力集为空时的出口：不硬猜一个分支，直接追问。 */
    private void handleClarify(String sessionId, String turnId, SseChannel out) {
        reply(sessionId, turnId, out,
                "抱歉，我没太理解您的具体诉求。您是想查询订单进度、了解售后政策，"
                        + "还是需要转接人工客服？",
                ErrorCode.ROUTE_LOW_CONFIDENCE);
    }

    /** 固定话术直接收口：写 reply + done 并落会话，不走生成。 */
    private void reply(String sessionId, String turnId, SseChannel out, String text) {
        reply(sessionId, turnId, out, text, null);
    }

    private void reply(String sessionId, String turnId, SseChannel out, String text, ErrorCode code) {
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "reply");
        event.put("text", text);
        if (code != null) {
            event.put("code", code.name());
        }
        writeEvent(out, event);
        writeEvent(out, Map.of("type", "done"));
        store.addAssistant(sessionId, turnId, text);
    }

    private void streamGenerate(
            String sessionId, String turnId, SseChannel out, List<Map<String, String>> messages) {
        long startedAt = System.nanoTime();
        AtomicBoolean firstToken = new AtomicBoolean(false);
        try (Stream<String> lines = client.generateStream(messages, MAX_NEW_TOKENS)) {
            lines.forEach(line -> {
                if ("token".equals(forwardLine(sessionId, turnId, out, line))
                        && firstToken.compareAndSet(false, true)) {
                    metrics.firstToken(Duration.ofNanos(System.nanoTime() - startedAt));
                }
            });
        } finally {
            metrics.generate(Duration.ofNanos(System.nanoTime() - startedAt));
        }
    }

    /**
     * 转发上游 SSE 行，但 error 事件必须换掉：上游事件里的 message 含底层异常，
     * 直接转发等于把内部细节透给用户（§9）。已输出的 token 保持原样。
     *
     * @return 事件类型（token / done / error）；非 data 行或解析失败返回空串
     */
    private String forwardLine(String sessionId, String turnId, SseChannel out, String line) {
        if (!line.startsWith("data: ")) {
            writeLine(out, line);
            return "";
        }
        JsonNode event;
        try {
            event = MAPPER.readTree(line.substring(6));
        } catch (IOException e) {
            writeLine(out, line);
            return "";
        }
        String type = event.path("type").asText("");
        if ("error".equals(type)) {
            ErrorCode code = ErrorCode.fromCode(event.path("code").asText(""));
            log.warn(
                    "upstream generation error session={} code={} partial_len={}",
                    sessionId,
                    code,
                    event.path("partial").asText("").length());
            writeError(out, code);
            return type;
        }
        writeLine(out, line);
        if ("done".equals(type)) {
            store.addAssistant(sessionId, turnId, event.path("reply").asText(null));
        }
        return type;
    }

    /** 写一条面向用户的错误事件；连接已断时静默放弃。 */
    private void writeError(SseChannel out, ErrorCode code) {
        try {
            writeEvent(out, ErrorEvents.payload(code));
        } catch (UncheckedIOException ignored) {
            log.debug("client disconnected while writing error {}", code);
        }
    }

    /**
     * 生成输入：系统提示 + 最近的对话 + 本轮证据与问题（§8 会话记忆）。
     *
     * <p>历史按原文逐条截断注入，让「它到哪了」这类指代有落点；本轮问题由控制器先落库，
     * 这里把它从历史里摘掉，避免同一句在上下文中出现两次。
     */
    private List<Map<String, String>> chatMessages(String sessionId, String context, String query) {
        String evidence = context.isBlank() ? "（本轮没有可用的订单或知识库上下文）" : context;
        String system = chatSystem.replace("{today}", LocalDate.now().toString());
        List<Map<String, String>> messages = new ArrayList<>();
        messages.add(Map.of("role", "system", "content", system));
        List<Map<String, String>> dialogue = recentDialogue(sessionId, query);
        messages.addAll(dialogue);
        String user = dialogue.isEmpty()
                ? evidence + "\n\n用户问题：" + query
                : evidence + "\n\n（以上早前对话仅作背景，与本轮资料冲突时以本轮资料为准）"
                        + "\n\n用户问题：" + query;
        messages.add(Map.of("role", "user", "content", user));
        return messages;
    }

    /** 上下文事件：前端据此展示本轮注入了哪些来源、是否发生冲突。 */
    private Map<String, Object> contextEvent(ContextAssembler.Assembled assembled) {
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "context");
        event.put("sources", assembled.sources());
        event.put("conflict", assembled.conflict());
        event.put("notes", assembled.notes());
        event.put("chars", assembled.text().length());
        return event;
    }

    /**
     * 闸门事件：三档判定 + 依据 + 阈值。分数只在本轮有召回时出现，
     * 不塞 null（SseChannel 的 ObjectMapper 不忽略 null 字段）。
     */
    private Map<String, Object> verdictEvent(AnswerabilityGate.Decision decision) {
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "verdict");
        event.put("verdict", decision.verdict().name().toLowerCase(Locale.ROOT));
        event.put("reason", decision.reason().name().toLowerCase(Locale.ROOT));
        if (decision.score() != null) {
            event.put("score", decision.score());
        }
        event.put("thresholds", Map.of(
                "low", AnswerabilityGate.T_LOW,
                "high", AnswerabilityGate.T_HIGH));
        return event;
    }

    /** 召回首位分数即候选里的最高分；没有召回时返回 null（线上同口径 §5.4）。 */
    private Double maxRecallScore(Knowledge knowledge) {
        if (knowledge == null || knowledge.candidates().isEmpty()) {
            return null;
        }
        return knowledge.candidates().stream()
                .mapToDouble(candidate -> ((Number) candidate.getOrDefault("score", 0.0)).doubleValue())
                .max()
                .getAsDouble();
    }

    /** 优先用意图模型抽到的订单号；没有就用正则从原句兜。都没有返回 null。 */
    private String resolveOrderId(String query, String hintedOrderId) {
        if (hintedOrderId != null && !hintedOrderId.isBlank()) {
            return hintedOrderId.trim();
        }
        Matcher matcher = ORDER_ID.matcher(query);
        return matcher.find() ? matcher.group() : null;
    }

    /** 按订单号精确查找，找不到返回 null（由调用方如实告知用户）。 */
    private OrderDetail findOrder(List<OrderDetail> orders, String orderId) {
        return orders.stream()
                .filter(order -> order.orderId().equalsIgnoreCase(orderId))
                .findFirst()
                .orElse(null);
    }

    /** 用户未指定订单号时取最近一笔；调用方已保证 orders 非空。 */
    private OrderDetail mostRecentOrder(List<OrderDetail> orders) {
        return orders.stream()
                .min(Comparator.comparingInt(OrderDetail::placedDaysAgo))
                .orElseThrow();
    }

    private Map<String, Object> orderSummary(OrderDetail order) {
        String items = order.items().stream()
                .map(item -> item.name() + "×" + item.quantity())
                .collect(Collectors.joining("、"));
        String latestTrace;
        if (order.logistics() != null && !order.logistics().traces().isEmpty()) {
            OrderDetail.LogisticsDetail.TraceDetail trace =
                    order.logistics().traces().get(order.logistics().traces().size() - 1);
            latestTrace = trace.date() + " " + trace.location() + " " + trace.desc();
        } else if (order.presale()) {
            latestTrace = "预售中，预计下单后 " + order.expectedShipDays() + " 天内发货";
        } else {
            latestTrace = "暂无物流信息";
        }
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("orderId", order.orderId());
        summary.put("statusLabel", order.statusLabel());
        summary.put("items", items);
        summary.put("latestTrace", latestTrace);
        summary.put("placedAt", order.placedAt());
        return summary;
    }

    /** 规划事件：能力集、主题、实体、门控与耗时一并推给前端，门控决策在前端可观测。 */
    private Map<String, Object> planEvent(CapabilityPlanner.Plan plan, String route) {
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "plan");
        event.put("route", route);
        event.put("capabilities", plan.capabilities());
        event.put("routeScores", plan.routeScores());
        Map<String, Object> topic = new LinkedHashMap<>();
        topic.put("label", plan.topic() == null ? "" : plan.topic());
        topic.put("confidence", plan.topicConfidence());
        event.put("topic", topic);
        event.put("entities", plan.entities());
        Map<String, Object> gate = new LinkedHashMap<>();
        plan.gate().forEach((name, decision) -> {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("allowed", decision.allowed());
            entry.put("reason", decision.reason());
            gate.put(name, entry);
        });
        event.put("gate", gate);
        event.put("source", plan.source());
        event.put("elapsed", plan.elapsedMs() / 1000.0);
        return event;
    }

    private String brief(String text) {
        return truncate(text, BRIEF_LIMIT);
    }

    private static String truncate(String text, int limit) {
        return text.length() <= limit ? text : text.substring(0, limit) + "…";
    }

    private static Duration seconds(double elapsedSec) {
        return Duration.ofMillis(Math.max(0, Math.round(elapsedSec * 1000)));
    }

    /** 订单能力结局：超时/异常（degraded）优先于业务上的查不到。 */
    private static String orderTag(FanOut.Outcome outcome, OrderLookup lookup) {
        if (!outcome.ok()) {
            return "degraded";
        }
        if (lookup instanceof OrderLookup.Missing) {
            return "missing";
        }
        if (lookup instanceof OrderLookup.NoContext) {
            return "no_context";
        }
        return "ok";
    }

    private static String read(Resource resource) {
        try {
            return resource.getContentAsString(StandardCharsets.UTF_8).trim();
        } catch (IOException e) {
            throw new IllegalStateException("加载 prompt 模板失败：" + resource, e);
        }
    }

    private void writeEvent(SseChannel out, Map<String, ?> payload) {
        out.send(payload);
    }

    private void writeLine(SseChannel out, String line) {
        out.sendLine(line);
    }
}
