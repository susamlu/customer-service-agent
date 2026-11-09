package com.example.csagent.chat;

import com.example.csagent.mock.model.OrderDetail;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.springframework.stereotype.Component;

/**
 * 上下文拼装：把并行能力的产出合并成一段生成输入（architecture.md §5.3）。
 *
 * <p>职责只有四条：去重、裁剪、来源标注、冲突提示。多能力合并会带来来源冲突，
 * 这是单标签路由不存在的风险，所以冲突必须显式告诉模型"以订单承诺为准"。
 */
@Component
public class ContextAssembler {

    // ml/experiments/eval_generation.py 用同名常量复刻这段拼装（§10.4）；改这里要同步改那边。
    private static final int KNOWLEDGE_MAX_ITEMS = 2;
    private static final int KNOWLEDGE_MAX_CHARS = 90;
    private static final int ORDER_MAX_CHARS = 200;

    /** 只认"N 天内送达/发货"这类时效承诺，避免把"7 天无理由"误判成时效冲突。 */
    private static final Pattern DELIVERY_PROMISE =
            Pattern.compile("(\\d+)\\s*(?:个)?(?:工作日|天)(?:内)?(?:送达|发货|到货)");

    private static final String CONFLICT_NOTE = "注意：订单承诺时效与知识库政策不一致，以订单承诺时效为准。";

    /** 查不到订单时要把"事实"告诉模型，而不是留空让它自由发挥（§8 铁律）。 */
    private static final String MISSING_ORDER_NOTE =
            "[订单] 用户提供的订单号在系统中不存在。不要编造该订单的任何信息。";

    private static final String NO_ORDER_NOTE =
            "[订单] 当前用户账号下没有任何订单记录。不要编造订单信息。";

    /**
     * 订单能力的四种结局。用枚举而不是布尔，是因为"查不到"有两种原因，
     * 而它们的正确话术不同：单号错 vs 没有订单。
     */
    public enum OrderState {
        /** 本轮没有订单能力，或订单已命中并正常渲染。 */
        NONE,
        /** 订单能力被请求了但执行失败（超时/异常）。 */
        UNAVAILABLE,
        /** 用户给的订单号在本人订单里查不到。 */
        MISSING,
        /** 用户账号下没有任何订单记录。 */
        NO_CONTEXT
    }

    /**
     * @param order     命中的订单；没有命中传 null
     * @param state     订单能力的结局，决定"查不到"时注入哪种事实
     * @param knowledge 重排后的知识片段，调用方保证已按相关度排序
     */
    public Assembled assemble(OrderDetail order, OrderState state, List<String> knowledge) {
        List<String> blocks = new ArrayList<>();
        List<String> sources = new ArrayList<>();
        List<String> notes = new ArrayList<>();

        String orderBlock = order == null ? null : limit(orderBlock(order), ORDER_MAX_CHARS);
        if (orderBlock != null) {
            blocks.add("[订单] " + orderBlock);
            sources.add("order");
        } else if (state == OrderState.UNAVAILABLE) {
            notes.add("订单信息暂时不可用");
        } else if (state == OrderState.MISSING) {
            blocks.add(MISSING_ORDER_NOTE);
            notes.add("订单号不存在");
        } else if (state == OrderState.NO_CONTEXT) {
            blocks.add(NO_ORDER_NOTE);
            notes.add("账号下没有订单记录");
        }

        Integer promisedDays = promisedDays(order);
        boolean conflict = false;
        Set<String> seen = new LinkedHashSet<>();
        int injected = 0;
        for (String raw : knowledge) {
            String snippet = raw == null ? "" : raw.strip();
            if (snippet.isEmpty() || !seen.add(snippet)) {
                continue;
            }
            if (promisedDays != null) {
                conflict |= contradicts(snippet, promisedDays);
            }
            blocks.add("[知识] " + limit(snippet, KNOWLEDGE_MAX_CHARS));
            if (++injected >= KNOWLEDGE_MAX_ITEMS) {
                break;
            }
        }
        if (injected > 0) {
            sources.add("knowledge");
        }
        if (conflict) {
            notes.add("订单承诺时效与知识库政策冲突");
            blocks.add(CONFLICT_NOTE);
        }

        return new Assembled(
                String.join("\n", blocks), List.copyOf(sources), List.copyOf(notes), conflict);
    }

    /** 压缩成一行：id+状态 → 最新物流 → 时效 → 商品与售后 → 实付。截断时先丢低价值信息。 */
    private String orderBlock(OrderDetail order) {
        List<String> parts = new ArrayList<>();
        parts.add(order.orderId() + " " + order.statusLabel() + "，" + order.placedDaysAgo() + " 天前下单");
        if (order.logistics() != null && !order.logistics().traces().isEmpty()) {
            OrderDetail.LogisticsDetail.TraceDetail latest =
                    order.logistics().traces().get(order.logistics().traces().size() - 1);
            parts.add("最新物流：" + latest.date() + " " + latest.location() + " " + latest.desc());
        } else {
            parts.add("暂无物流");
        }
        if (order.presale()) {
            parts.add("预售预计 " + order.expectedShipDays() + " 天内发货");
        }
        if (order.promisedDays() != null && order.promisedDays() > 0) {
            parts.add("承诺 " + order.promisedDays() + " 天内送达");
        }
        for (OrderDetail.ItemDetail item : order.items()) {
            String guard = item.sevenDayReturn() ? "支持七天无理由" : "不支持七天无理由";
            if (item.supportsPriceProtection()) {
                guard += "，价保 " + item.priceProtectionDays() + " 天";
            }
            parts.add(item.name() + "×" + item.quantity() + "（" + guard + "）");
        }
        parts.add("实付 " + String.format("%.2f", order.totalAmount()) + " 元");
        return String.join("；", parts);
    }

    /** 订单侧承诺的天数：优先正式时效，预售退化到预计发货天数。 */
    private Integer promisedDays(OrderDetail order) {
        if (order == null) {
            return null;
        }
        if (order.promisedDays() != null && order.promisedDays() > 0) {
            return order.promisedDays();
        }
        return order.presale() ? order.expectedShipDays() : null;
    }

    private boolean contradicts(String snippet, int promisedDays) {
        Matcher matcher = DELIVERY_PROMISE.matcher(snippet);
        while (matcher.find()) {
            if (Integer.parseInt(matcher.group(1)) != promisedDays) {
                return true;
            }
        }
        return false;
    }

    private String limit(String text, int max) {
        return text.length() <= max ? text : text.substring(0, max - 1) + "…";
    }

    /** 拼好的生成上下文；text 为空表示所有能力都没有产出。 */
    public record Assembled(
            String text, List<String> sources, List<String> notes, boolean conflict) {

        public boolean empty() {
            return text.isBlank();
        }
    }
}
