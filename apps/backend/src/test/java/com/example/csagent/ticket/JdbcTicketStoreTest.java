package com.example.csagent.ticket;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.UUID;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.jdbc.datasource.init.ResourceDatabasePopulator;

/** JDBC 工单存储（§6）：建单、查询、状态机闸门、条件更新的并发语义。 */
class JdbcTicketStoreTest {

    private JdbcTemplate jdbc;
    private JdbcTicketStore store;

    @BeforeEach
    void setUp() {
        // 每个测试一条独立内存库，互不串味。建表脚本读 classpath 上那一份，
        // 测试跑的是和生产完全相同的 schema，不用另维护一份。
        DriverManagerDataSource dataSource = new DriverManagerDataSource(
                "jdbc:h2:mem:ticket-" + UUID.randomUUID() + ";DB_CLOSE_DELAY=-1", "sa", "");
        new ResourceDatabasePopulator(new ClassPathResource("schema.sql")).execute(dataSource);
        jdbc = new JdbcTemplate(dataSource);
        store = new JdbcTicketStore(jdbc);
    }

    @Test
    @DisplayName("create stores a pending ticket with timestamps")
    void createStartsPending() {
        Ticket ticket = store.create("s1", "u1", "我要转人工", "user: 你好");

        assertThat(ticket.id()).isPositive();
        assertThat(ticket.status()).isEqualTo(TicketStatus.PENDING);
        assertThat(ticket.sessionId()).isEqualTo("s1");
        assertThat(ticket.userId()).isEqualTo("u1");
        assertThat(ticket.context()).isEqualTo("user: 你好");
        assertThat(ticket.createdAt()).isEqualTo(ticket.updatedAt());
    }

    @Test
    @DisplayName("a null context round-trips as null")
    void createAcceptsNullContext() {
        Ticket ticket = store.create("s1", "u1", "q", null);

        assertThat(store.find(ticket.id()).orElseThrow().context()).isNull();
    }

    @Test
    @DisplayName("find returns empty for an unknown id")
    void findUnknownIsEmpty() {
        assertThat(store.find(999)).isEmpty();
    }

    @Test
    @DisplayName("list filters by status and returns newest first")
    void listFiltersAndOrders() {
        Ticket first = store.create("s1", "u1", "a", null);
        Ticket second = store.create("s2", "u1", "b", null);
        store.transition(first.id(), TicketStatus.IN_PROGRESS);

        assertThat(store.list(null))
                .extracting(Ticket::id)
                .containsExactly(second.id(), first.id());
        assertThat(store.list(TicketStatus.PENDING))
                .extracting(Ticket::id)
                .containsExactly(second.id());
        assertThat(store.list(TicketStatus.IN_PROGRESS))
                .extracting(Ticket::id)
                .containsExactly(first.id());
    }

    @Test
    @DisplayName("findOpen only returns unfinished tickets of that session")
    void findOpenIgnoresFinishedTickets() {
        Ticket closed = store.create("s1", "u1", "a", null);
        store.transition(closed.id(), TicketStatus.CLOSED);

        assertThat(store.findOpen("s1")).isEmpty();

        Ticket open = store.create("s1", "u1", "b", null);

        assertThat(store.findOpen("s1")).map(Ticket::id).contains(open.id());
        assertThat(store.findOpen("s2")).isEmpty();
    }

    @Test
    @DisplayName("the full happy path walks pending -> in progress -> resolved")
    void happyPath() {
        Ticket ticket = store.create("s1", "u1", "q", null);

        Ticket accepted = store.transition(ticket.id(), TicketStatus.IN_PROGRESS);
        Ticket resolved = store.transition(ticket.id(), TicketStatus.RESOLVED);

        assertThat(accepted.status()).isEqualTo(TicketStatus.IN_PROGRESS);
        assertThat(resolved.status()).isEqualTo(TicketStatus.RESOLVED);
        assertThat(resolved.updatedAt()).isGreaterThanOrEqualTo(accepted.updatedAt());
    }

    @Test
    @DisplayName("transitioning a terminal ticket is rejected")
    void terminalTicketRejectsTransition() {
        Ticket ticket = store.create("s1", "u1", "q", null);
        store.transition(ticket.id(), TicketStatus.CLOSED);

        assertThatThrownBy(() -> store.transition(ticket.id(), TicketStatus.IN_PROGRESS))
                .isInstanceOf(TicketTransitionException.class)
                .hasMessageContaining("CLOSED");
    }

    @Test
    @DisplayName("skipping straight to resolved is rejected")
    void pendingCannotResolveDirectly() {
        Ticket ticket = store.create("s1", "u1", "q", null);

        assertThatThrownBy(() -> store.transition(ticket.id(), TicketStatus.RESOLVED))
                .isInstanceOf(TicketTransitionException.class);
    }

    @Test
    @DisplayName("transitioning an unknown id reports not found")
    void unknownTicketTransitionFails() {
        assertThatThrownBy(() -> store.transition(4242, TicketStatus.CLOSED))
                .isInstanceOf(TicketNotFoundException.class)
                .hasMessageContaining("4242");
    }

    @Test
    @DisplayName("a status changed underneath the caller surfaces as an invalid transition, not a silent overwrite")
    void concurrentChangeIsDetectedByConditionalUpdate() {
        Ticket ticket = store.create("s1", "u1", "q", null);
        // 模拟另一个客服抢先处理：直接改库，绕过 store 的内存校验。
        jdbc.update("UPDATE ticket SET status = 'RESOLVED' WHERE id = ?", ticket.id());

        assertThatThrownBy(() -> store.transition(ticket.id(), TicketStatus.IN_PROGRESS))
                .isInstanceOf(TicketTransitionException.class)
                .hasMessageContaining("RESOLVED");

        assertThat(store.find(ticket.id()).orElseThrow().status())
                .isEqualTo(TicketStatus.RESOLVED);
    }

    @Test
    @DisplayName("list is bounded so a large backlog cannot exhaust memory")
    void listIsBounded() {
        for (int i = 0; i < 205; i++) {
            store.create("s" + i, "u1", "q", null);
        }

        List<Ticket> tickets = store.list(null);

        assertThat(tickets).hasSize(200);
    }
}
