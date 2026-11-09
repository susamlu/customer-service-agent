package com.example.csagent.ticket;

/** 非法状态迁移（如已关闭的工单再接受）。REST 层映射成 409 + {@code TICKET_TRANSITION_INVALID}。 */
public class TicketTransitionException extends RuntimeException {

    private final long ticketId;
    private final TicketStatus from;
    private final TicketStatus to;

    public TicketTransitionException(long ticketId, TicketStatus from, TicketStatus to) {
        super("工单状态迁移非法: id=" + ticketId + " " + from + " -> " + to);
        this.ticketId = ticketId;
        this.from = from;
        this.to = to;
    }

    public long ticketId() {
        return ticketId;
    }

    public TicketStatus from() {
        return from;
    }

    public TicketStatus to() {
        return to;
    }
}
