package com.example.csagent.ticket;

/** 工单不存在。REST 层映射成 404 + {@code TICKET_NOT_FOUND}。 */
public class TicketNotFoundException extends RuntimeException {

    private final long ticketId;

    public TicketNotFoundException(long ticketId) {
        super("工单不存在: id=" + ticketId);
        this.ticketId = ticketId;
    }

    public long ticketId() {
        return ticketId;
    }
}
