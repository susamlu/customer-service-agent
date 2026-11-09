package com.example.csagent.ticket;

/**
 * 转人工工单（architecture.md §6）。
 *
 * <p>{@code context} 存转人工那一刻的上下文快照（订单/知识证据摘要），
 * 让接手的人工客服不用重放整段会话也能知道用户到底在问什么。
 */
public record Ticket(
        long id,
        String sessionId,
        String userId,
        TicketStatus status,
        String query,
        String context,
        long createdAt,
        long updatedAt) {}
