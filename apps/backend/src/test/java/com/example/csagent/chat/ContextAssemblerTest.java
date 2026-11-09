package com.example.csagent.chat;

import static org.assertj.core.api.Assertions.assertThat;

import com.example.csagent.mock.model.OrderDetail;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** 上下文拼装（§5.3）：去重、裁剪、来源标注、冲突提示。 */
class ContextAssemblerTest {

    private final ContextAssembler assembler = new ContextAssembler();

    private static OrderDetail.ItemDetail item(String name) {
        return new OrderDetail.ItemDetail(
                "P1", name, "服装", 199.0, 1, true, 15, true, "吊牌完好");
    }

    private static OrderDetail order(
            Integer promisedDays, boolean presale, Integer expectedShipDays,
            List<OrderDetail.LogisticsDetail.TraceDetail> traces, List<OrderDetail.ItemDetail> items) {
        return new OrderDetail(
                "SO123456", "SHIPPED", "已发货", "2026-11-01", 7, 199.0, "支付宝",
                promisedDays, presale, expectedShipDays, "",
                items,
                new OrderDetail.LogisticsDetail("顺丰", "SF123", traces));
    }

    private static OrderDetail sampleOrder(Integer promisedDays) {
        return order(
                promisedDays, false, null,
                List.of(new OrderDetail.LogisticsDetail.TraceDetail(
                        "2026-11-05", 3, "杭州", "已揽收")),
                List.of(item("卫衣")));
    }

    @Test
    @DisplayName("no capabilities produce a blank context")
    void nothingAvailable_isEmpty() {
        ContextAssembler.Assembled result =
                assembler.assemble(null, ContextAssembler.OrderState.NONE, List.of());

        assertThat(result.text()).isBlank();
        assertThat(result.empty()).isTrue();
        assertThat(result.sources()).isEmpty();
        assertThat(result.notes()).isEmpty();
        assertThat(result.conflict()).isFalse();
    }

    @Test
    @DisplayName("a degraded order capability is noted but does not block generation")
    void orderUnavailable_addsNote() {
        ContextAssembler.Assembled result =
                assembler.assemble(null, ContextAssembler.OrderState.UNAVAILABLE, List.of());

        assertThat(result.text()).isBlank();
        assertThat(result.notes()).contains("订单信息暂时不可用");
        assertThat(result.sources()).isEmpty();
    }

    @Test
    @DisplayName("a missing order is injected as a fact so the model cannot invent logistics")
    void missingOrder_injectsFact() {
        ContextAssembler.Assembled result =
                assembler.assemble(null, ContextAssembler.OrderState.MISSING, List.of("七天无理由"));

        assertThat(result.text())
                .startsWith("[订单] 用户提供的订单号在系统中不存在")
                .contains("[知识] 七天无理由");
        assertThat(result.notes()).contains("订单号不存在");
        assertThat(result.sources()).containsExactly("knowledge");
    }

    @Test
    @DisplayName("an account with no orders is injected as a fact, not left blank")
    void noOrderContext_injectsFact() {
        ContextAssembler.Assembled result =
                assembler.assemble(null, ContextAssembler.OrderState.NO_CONTEXT, List.of());

        assertThat(result.text()).contains("[订单] 当前用户账号下没有任何订单记录");
        assertThat(result.notes()).contains("账号下没有订单记录");
        assertThat(result.sources()).isEmpty();
    }

    @Test
    @DisplayName("order summary carries id, status, latest trace, promise and amount")
    void orderPresent_rendersSummary() {
        ContextAssembler.Assembled result =
                assembler.assemble(sampleOrder(3), ContextAssembler.OrderState.NONE, List.of());

        assertThat(result.text())
                .startsWith("[订单] SO123456 已发货，7 天前下单")
                .contains("最新物流：2026-11-05 杭州 已揽收")
                .contains("承诺 3 天内送达")
                .contains("卫衣×1（支持七天无理由，价保 15 天）")
                .contains("实付 199.00 元");
        assertThat(result.sources()).containsExactly("order");
        assertThat(result.notes()).isEmpty();
        assertThat(result.conflict()).isFalse();
    }

    @Test
    @DisplayName("missing logistics renders a placeholder instead of an empty field")
    void orderWithoutTraces_usesPlaceholder() {
        OrderDetail order = order(null, false, null, List.of(), List.of(item("卫衣")));

        String text = assembler.assemble(order, ContextAssembler.OrderState.NONE, List.of()).text();

        assertThat(text).contains("暂无物流");
    }

    @Test
    @DisplayName("presale order falls back to expected ship days")
    void presaleOrder_usesExpectedShipDays() {
        OrderDetail order = order(null, true, 20, List.of(), List.of(item("卫衣")));

        ContextAssembler.Assembled result =
                assembler.assemble(order, ContextAssembler.OrderState.NONE, List.of("15 天内发货"));

        assertThat(result.text()).contains("预售预计 20 天内发货");
        assertThat(result.conflict()).isTrue();
    }

    @Test
    @DisplayName("knowledge snippets are deduped, trimmed and capped at two")
    void knowledge_dedupesTrimsAndCaps() {
        ContextAssembler.Assembled result = assembler.assemble(
                null, ContextAssembler.OrderState.NONE, List.of(" 七天无理由 ", "七天无理由", "", "   ", "第三条"));

        assertThat(result.text()).isEqualTo("[知识] 七天无理由\n[知识] 第三条");
        assertThat(result.sources()).containsExactly("knowledge");
    }

    @Test
    @DisplayName("knowledge snippet is truncated at 90 chars with an ellipsis")
    void knowledge_truncated() {
        String longSnippet = "政".repeat(120);

        String text =
                assembler.assemble(null, ContextAssembler.OrderState.NONE, List.of(longSnippet))
                        .text();

        String block = text.substring("[知识] ".length());
        assertThat(block).hasSize(90).endsWith("…");
    }

    @Test
    @DisplayName("a contradicting delivery promise is surfaced explicitly")
    void conflict_isDetectedAndNoted() {
        ContextAssembler.Assembled result =
                assembler.assemble(sampleOrder(3), ContextAssembler.OrderState.NONE, List.of("我们承诺 5 天内送达"));

        assertThat(result.conflict()).isTrue();
        assertThat(result.notes()).contains("订单承诺时效与知识库政策冲突");
        assertThat(result.text()).contains("以订单承诺时效为准");
    }

    @Test
    @DisplayName("a matching promise is not a conflict")
    void matchingPromise_noConflict() {
        ContextAssembler.Assembled result =
                assembler.assemble(sampleOrder(3), ContextAssembler.OrderState.NONE, List.of("3 天内送达"));

        assertThat(result.conflict()).isFalse();
        assertThat(result.notes()).isEmpty();
        assertThat(result.text()).doesNotContain("以订单承诺时效为准");
    }

    @Test
    @DisplayName("seven-day return policy is not mistaken for a delivery promise")
    void sevenDayReturn_isNotADeliveryConflict() {
        ContextAssembler.Assembled result =
                assembler.assemble(sampleOrder(3), ContextAssembler.OrderState.NONE, List.of("支持7天无理由退货"));

        assertThat(result.conflict()).isFalse();
    }

    @Test
    @DisplayName("sources list order before knowledge")
    void sources_followInjectionOrder() {
        ContextAssembler.Assembled result =
                assembler.assemble(sampleOrder(3), ContextAssembler.OrderState.NONE, List.of("七天无理由"));

        assertThat(result.sources()).containsExactly("order", "knowledge");
    }

    @Test
    @DisplayName("an oversized order block is truncated to 200 chars")
    void orderBlock_truncated() {
        List<OrderDetail.ItemDetail> items = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            items.add(item("超长商品名称".repeat(6)));
        }
        OrderDetail order = order(
                null, false, null,
                List.of(new OrderDetail.LogisticsDetail.TraceDetail(
                        "2026-11-05", 3, "杭州", "已揽收")),
                items);

        String text = assembler.assemble(order, ContextAssembler.OrderState.NONE, List.of()).text();

        assertThat(text).hasSizeLessThanOrEqualTo("[订单] ".length() + 200).endsWith("…");
    }
}
