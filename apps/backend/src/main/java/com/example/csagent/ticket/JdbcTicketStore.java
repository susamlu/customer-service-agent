package com.example.csagent.ticket;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.List;
import java.util.Optional;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.support.GeneratedKeyHolder;
import org.springframework.jdbc.support.KeyHolder;
import org.springframework.stereotype.Repository;

/**
 * 工单的 JDBC 实现（architecture.md §6）。
 *
 * <p>状态迁移用 {@code UPDATE ... WHERE id=? AND status=?} 做条件更新：并发下两个客服
 * 同时点「接受」，只有一个能改成功，另一个拿到 409 而不是把状态覆盖掉。校验先在内存
 * 做一次是为了给出准确的 from/to，真正的并发闸门在 SQL 的 WHERE 上。
 */
@Repository
public class JdbcTicketStore implements TicketStore {

    /** 列表不做分页参数，先给一个上限，避免无界查询（真实产品接后台时再补游标）。 */
    private static final int LIST_LIMIT = 200;

    private static final String COLUMNS =
            "id, session_id, user_id, status, query, context, created_at, updated_at";

    private static final RowMapper<Ticket> ROW_MAPPER = (ResultSet rs, int rowNum) -> new Ticket(
            rs.getLong("id"),
            rs.getString("session_id"),
            rs.getString("user_id"),
            TicketStatus.valueOf(rs.getString("status")),
            rs.getString("query"),
            rs.getString("context"),
            rs.getLong("created_at"),
            rs.getLong("updated_at"));

    private final JdbcTemplate jdbc;

    public JdbcTicketStore(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    @Override
    public Ticket create(String sessionId, String userId, String query, String context) {
        long now = System.currentTimeMillis();
        KeyHolder keys = new GeneratedKeyHolder();
        jdbc.update(connection -> {
            PreparedStatement ps = connection.prepareStatement(
                    "INSERT INTO ticket"
                            + " (session_id, user_id, status, query, context, created_at, updated_at)"
                            + " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    new String[] {"id"});
            ps.setString(1, sessionId);
            ps.setString(2, userId);
            ps.setString(3, TicketStatus.PENDING.name());
            ps.setString(4, query);
            ps.setString(5, context);
            ps.setLong(6, now);
            ps.setLong(7, now);
            return ps;
        }, keys);
        Number id = keys.getKey();
        if (id == null) {
            throw new IllegalStateException("工单插入未返回主键");
        }
        return find(id.longValue()).orElseThrow(() -> new IllegalStateException("工单写入后读不到"));
    }

    @Override
    public Optional<Ticket> find(long id) {
        return jdbc.query("SELECT " + COLUMNS + " FROM ticket WHERE id = ?", ROW_MAPPER, id)
                .stream()
                .findFirst();
    }

    @Override
    public List<Ticket> list(TicketStatus status) {
        if (status == null) {
            return jdbc.query(
                    "SELECT " + COLUMNS + " FROM ticket ORDER BY id DESC LIMIT " + LIST_LIMIT,
                    ROW_MAPPER);
        }
        return jdbc.query(
                "SELECT " + COLUMNS + " FROM ticket WHERE status = ? ORDER BY id DESC LIMIT "
                        + LIST_LIMIT,
                ROW_MAPPER,
                status.name());
    }

    @Override
    public Optional<Ticket> findOpen(String sessionId) {
        return jdbc.query(
                        "SELECT " + COLUMNS + " FROM ticket"
                                + " WHERE session_id = ? AND status IN ('PENDING', 'IN_PROGRESS')"
                                + " ORDER BY id DESC LIMIT 1",
                        ROW_MAPPER,
                        sessionId)
                .stream()
                .findFirst();
    }

    @Override
    public Ticket transition(long id, TicketStatus target) {
        Ticket current = find(id).orElseThrow(() -> new TicketNotFoundException(id));
        if (!current.status().canTransitionTo(target)) {
            throw new TicketTransitionException(id, current.status(), target);
        }
        int updated = jdbc.update(
                "UPDATE ticket SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                target.name(),
                System.currentTimeMillis(),
                id,
                current.status().name());
        if (updated == 0) {
            // 并发的另一个请求抢先改了状态：按最新状态重判，非法则报 409，被删则报 404。
            Ticket latest = find(id).orElseThrow(() -> new TicketNotFoundException(id));
            throw new TicketTransitionException(id, latest.status(), target);
        }
        return find(id).orElseThrow(() -> new TicketNotFoundException(id));
    }
}
