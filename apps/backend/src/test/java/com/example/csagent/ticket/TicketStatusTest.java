package com.example.csagent.ticket;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** 工单状态机（§6）：只走允许的边，终态没有出口。 */
class TicketStatusTest {

    @Test
    @DisplayName("pending can be accepted or closed, never resolved directly")
    void pendingTransitions() {
        assertThat(TicketStatus.PENDING.canTransitionTo(TicketStatus.IN_PROGRESS)).isTrue();
        assertThat(TicketStatus.PENDING.canTransitionTo(TicketStatus.CLOSED)).isTrue();
        assertThat(TicketStatus.PENDING.canTransitionTo(TicketStatus.RESOLVED)).isFalse();
        assertThat(TicketStatus.PENDING.canTransitionTo(TicketStatus.PENDING)).isFalse();
    }

    @Test
    @DisplayName("in progress can be resolved or closed")
    void inProgressTransitions() {
        assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.RESOLVED)).isTrue();
        assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.CLOSED)).isTrue();
        assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.PENDING)).isFalse();
        assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.IN_PROGRESS)).isFalse();
    }

    @Test
    @DisplayName("terminal states have no outgoing transitions")
    void terminalStatesAreClosed() {
        for (TicketStatus terminal : new TicketStatus[] {TicketStatus.RESOLVED, TicketStatus.CLOSED}) {
            for (TicketStatus target : TicketStatus.values()) {
                assertThat(terminal.canTransitionTo(target))
                        .as("%s -> %s", terminal, target)
                        .isFalse();
            }
        }
    }

    @Test
    @DisplayName("parse ignores case and surrounding whitespace")
    void parseIsTolerant() {
        assertThat(TicketStatus.parse(" in_progress ")).isEqualTo(TicketStatus.IN_PROGRESS);
        assertThat(TicketStatus.parse("Closed")).isEqualTo(TicketStatus.CLOSED);
    }

    @Test
    @DisplayName("parse returns null for unknown or missing input")
    void parseRejectsUnknown() {
        assertThat(TicketStatus.parse("archived")).isNull();
        assertThat(TicketStatus.parse(null)).isNull();
    }
}
