package com.example.csagent.chat;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** 会话仓库：按轮归档意图（并发两轮不互相覆盖）、消息封顶、空闲淘汰。 */
class SessionStoreTest {

    private final SessionStore store = new SessionStore();

    @Test
    @DisplayName("unknown session has no history")
    void unknownSession_emptyHistory() {
        assertThat(store.history("nope")).isEmpty();
    }

    @Test
    @DisplayName("assistant message carries the route and domain recorded for that turn")
    void assistantCarriesTurnIntent() {
        store.recordIntent("turn-1", "order", "物流");

        store.addAssistant("s1", "turn-1", "你的包裹在派送中");

        assertThat(store.history("s1")).singleElement().satisfies(message -> {
            assertThat(message.role()).isEqualTo("assistant");
            assertThat(message.route()).isEqualTo("order");
            assertThat(message.domain()).isEqualTo("物流");
        });
    }

    @Test
    @DisplayName("concurrent turns keep their own intent, arrival order irrelevant")
    void concurrentTurns_doNotOverwriteEachOther() {
        // 旧实现按 sessionId 存待归档意图：两轮同一会话时后一轮会覆盖前一轮。
        // 按 turnId 归档后，谁先落库都不会串味。
        store.recordIntent("turn-a", "knowledge", "退换货");
        store.recordIntent("turn-b", "order", "物流");

        store.addAssistant("s1", "turn-b", "订单答复");
        store.addAssistant("s1", "turn-a", "知识答复");

        assertThat(store.history("s1"))
                .extracting(ChatMessage::route)
                .containsExactly("order", "knowledge");
    }

    @Test
    @DisplayName("assistant message without a turn id has no intent attached")
    void assistantWithoutTurnId_hasNoIntent() {
        store.addAssistant("s1", null, "答复");

        assertThat(store.history("s1")).singleElement().satisfies(message -> {
            assertThat(message.route()).isNull();
            assertThat(message.domain()).isNull();
        });
    }

    @Test
    @DisplayName("blank assistant text is not stored")
    void blankAssistantText_isIgnored() {
        store.addAssistant("s1", "turn-1", "   ");

        assertThat(store.history("s1")).isEmpty();
    }

    @Test
    @DisplayName("history is capped at 200 messages, oldest dropped first")
    void history_isCapped() {
        for (int i = 0; i < 205; i++) {
            store.addUser("s1", "msg-" + i);
        }

        List<ChatMessage> history = store.history("s1");

        assertThat(history).hasSize(200);
        assertThat(history.get(0).text()).isEqualTo("msg-5");
        assertThat(history.get(199).text()).isEqualTo("msg-204");
    }

    @Test
    @DisplayName("idle sessions are evicted once the TTL passes")
    void idleSession_evictedAfterTtl() {
        store.addUser("s1", "hi");

        store.sweep(System.currentTimeMillis() + SessionStore.IDLE_TTL_MILLIS + 1);

        assertThat(store.history("s1")).isEmpty();
    }

    @Test
    @DisplayName("an expired pending intent does not attach to a late reply")
    void expiredPendingIntent_isNotAttached() {
        store.recordIntent("turn-1", "order", "物流");

        store.sweep(System.currentTimeMillis() + SessionStore.IDLE_TTL_MILLIS + 1);
        store.addAssistant("s1", "turn-1", "迟到的答复");

        assertThat(store.history("s1")).singleElement().satisfies(message -> {
            assertThat(message.text()).isEqualTo("迟到的答复");
            assertThat(message.route()).isNull();
        });
    }

    @Test
    @DisplayName("clear removes the session's history")
    void clear_removesHistory() {
        store.addUser("s1", "hi");

        store.clear("s1");

        assertThat(store.history("s1")).isEmpty();
    }
}
