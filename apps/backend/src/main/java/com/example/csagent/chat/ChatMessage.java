package com.example.csagent.chat;

/** 一条会话消息。assistant 消息附带当次的一级意图与知识主题。 */
public record ChatMessage(
        String role,
        String text,
        String route,
        String domain,
        long createdAt
) {

    public static ChatMessage user(String text) {
        return new ChatMessage("user", text, null, null, System.currentTimeMillis());
    }

    public static ChatMessage assistant(String text, String route, String domain) {
        return new ChatMessage("assistant", text, route, domain, System.currentTimeMillis());
    }
}
