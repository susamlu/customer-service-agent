package com.example.csagent.ticket;

import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * 工单状态机（architecture.md §6）。
 *
 * <pre>
 *   PENDING ──accept──▶ IN_PROGRESS ──resolve──▶ RESOLVED（终态）
 *      │                     │
 *      └──────close──────────┴──close──▶ CLOSED（终态）
 * </pre>
 *
 * <p>终态不设出口：已解决的工单再被关闭只会掩盖真实结论，宁可让调用方拿到 409。
 * 允许的迁移写死在这里，存储层只做条件更新，两边不各判一套。
 */
public enum TicketStatus {
    PENDING,
    IN_PROGRESS,
    RESOLVED,
    CLOSED;

    private static final Map<TicketStatus, Set<TicketStatus>> ALLOWED = Map.of(
            PENDING, Set.of(IN_PROGRESS, CLOSED),
            IN_PROGRESS, Set.of(RESOLVED, CLOSED),
            RESOLVED, Set.of(),
            CLOSED, Set.of());

    public boolean canTransitionTo(TicketStatus target) {
        return ALLOWED.get(this).contains(target);
    }

    /** 解析外部传入的状态串；无法识别时返回 null，由调用方决定报什么错。 */
    public static TicketStatus parse(String raw) {
        if (raw == null) {
            return null;
        }
        for (TicketStatus candidate : values()) {
            if (candidate.name().equalsIgnoreCase(raw.trim())) {
                return candidate;
            }
        }
        return null;
    }
}
