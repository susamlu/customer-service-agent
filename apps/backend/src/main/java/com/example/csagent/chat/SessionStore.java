package com.example.csagent.chat;

import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.springframework.stereotype.Component;

/**
 * 内存会话仓库。演示用途，重启即清空。
 *
 * <p>两条并发约束：
 *
 * <ul>
 *   <li>意图归档按 turnId 索引而不是 sessionId：同一会话并发两轮时，后一轮不会覆盖
 *       前一轮待归档的 route（否则两轮回复会挂上同一个意图）。
 *   <li>会话与待归档意图都有空闲 TTL，惰性清理，避免 map 随会话数无限增长。
 * </ul>
 */
@Component
public class SessionStore {

    private static final int MAX_MESSAGES = 200;
    /** 会话空闲超过这个时长即淘汰；进程内演示取值，生产切 Redis 后由 TTL 策略接管。 */
    static final long IDLE_TTL_MILLIS = Duration.ofMinutes(30).toMillis();
    /** 清理的最小间隔：每次访问都全表扫描会白烧 CPU，一分钟一次足够。 */
    private static final long SWEEP_INTERVAL_MILLIS = Duration.ofMinutes(1).toMillis();

    private final Map<String, Session> sessions = new ConcurrentHashMap<>();
    /** 本轮请求已识别的一级意图与知识主题，按 turnId 归档，回复落地时取走。 */
    private final Map<String, PendingIntent> pending = new ConcurrentHashMap<>();
    private final AtomicLong nextSweepAtMillis =
            new AtomicLong(System.currentTimeMillis() + SWEEP_INTERVAL_MILLIS);

    public void addUser(String sessionId, String text) {
        append(sessionId, ChatMessage.user(text));
    }

    public void addAssistant(String sessionId, String turnId, String text) {
        if (text == null || text.isBlank()) {
            return;
        }
        PendingIntent intent = turnId == null ? null : pending.remove(turnId);
        ChatMessage message = intent == null
                ? ChatMessage.assistant(text, null, null)
                : ChatMessage.assistant(text, intent.route(), intent.domain());
        append(sessionId, message);
    }

    public void recordIntent(String turnId, String route, String domain) {
        pending.put(turnId, new PendingIntent(
                route,
                domain == null || domain.isBlank() ? null : domain,
                System.currentTimeMillis()));
    }

    public List<ChatMessage> history(String sessionId) {
        maybeSweep();
        Session session = sessions.get(sessionId);
        if (session == null) {
            return List.of();
        }
        session.touch();
        synchronized (session.messages) {
            return new ArrayList<>(session.messages);
        }
    }

    public void clear(String sessionId) {
        sessions.remove(sessionId);
    }

    /** 淘汰空闲会话与过期意图；包可见，测试可直接按指定时刻触发。 */
    void sweep(long nowMillis) {
        sessions.entrySet().removeIf(
                entry -> nowMillis - entry.getValue().lastAccessMillis > IDLE_TTL_MILLIS);
        pending.entrySet().removeIf(
                entry -> nowMillis - entry.getValue().createdAtMillis() > IDLE_TTL_MILLIS);
    }

    private void maybeSweep() {
        long now = System.currentTimeMillis();
        long due = nextSweepAtMillis.get();
        if (now < due || !nextSweepAtMillis.compareAndSet(due, now + SWEEP_INTERVAL_MILLIS)) {
            return;
        }
        sweep(now);
    }

    private void append(String sessionId, ChatMessage message) {
        maybeSweep();
        Session session = sessions.computeIfAbsent(sessionId, key -> new Session());
        session.touch();
        synchronized (session.messages) {
            session.messages.addLast(message);
            while (session.messages.size() > MAX_MESSAGES) {
                session.messages.removeFirst();
            }
        }
    }

    private record PendingIntent(String route, String domain, long createdAtMillis) {}

    private static final class Session {
        private final Deque<ChatMessage> messages = new ArrayDeque<>();
        private volatile long lastAccessMillis = System.currentTimeMillis();

        void touch() {
            lastAccessMillis = System.currentTimeMillis();
        }
    }
}
