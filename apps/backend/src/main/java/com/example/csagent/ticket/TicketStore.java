package com.example.csagent.ticket;

import java.util.List;
import java.util.Optional;

/**
 * 工单存储（architecture.md §6）。接口化是为了让测试用内存实现、生产用 JDBC，
 * 也让后续换库只改实现。
 */
public interface TicketStore {

    /** 新建一条待处理工单。{@code context} 可为 null。 */
    Ticket create(String sessionId, String userId, String query, String context);

    Optional<Ticket> find(long id);

    /** {@code status} 为 null 时返回全部，按创建时间倒序。 */
    List<Ticket> list(TicketStatus status);

    /**
     * 同一会话还没结束的工单（PENDING 或 IN_PROGRESS），用于避免重复转人工建单。
     */
    Optional<Ticket> findOpen(String sessionId);

    /**
     * 状态迁移。工单不存在抛 {@link TicketNotFoundException}，
     * 当前状态不允许迁到 {@code target} 抛 {@link TicketTransitionException}。
     */
    Ticket transition(long id, TicketStatus target);
}
