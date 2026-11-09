package com.example.csagent.ticket;

/** 列表过滤传了不认识的状态串。REST 层映射成 400 + {@code INVALID_REQUEST}。 */
public class InvalidTicketStatusException extends RuntimeException {

    private final String raw;

    public InvalidTicketStatusException(String raw) {
        super("未知工单状态: " + raw);
        this.raw = raw;
    }

    public String raw() {
        return raw;
    }
}
