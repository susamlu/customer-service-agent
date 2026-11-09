package com.example.csagent.ticket;

import com.example.csagent.chat.ErrorCode;
import com.example.csagent.chat.ErrorEvents;
import java.util.List;
import java.util.Map;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 人工客服工单接口（architecture.md §6）。
 *
 * <pre>
 *   GET  /api/tickets/{id}        → 单条
 *   GET  /api/tickets?status=     → 列表（省略 status 返回全部）
 *   POST /api/tickets/{id}/accept → PENDING → IN_PROGRESS
 *   POST /api/tickets/{id}/resolve→ IN_PROGRESS → RESOLVED
 *   POST /api/tickets/{id}/close  → PENDING / IN_PROGRESS → CLOSED
 * </pre>
 *
 * <p>非法流转返回 409，工单不存在返回 404，错误体与 SSE 事件同形（§3.5）。
 */
@RestController
@RequestMapping("/api/tickets")
public class TicketController {

    private final TicketStore store;

    public TicketController(TicketStore store) {
        this.store = store;
    }

    @GetMapping("/{id}")
    public Ticket get(@PathVariable long id) {
        return store.find(id).orElseThrow(() -> new TicketNotFoundException(id));
    }

    @GetMapping
    public List<Ticket> list(@RequestParam(required = false) String status) {
        if (status == null || status.isBlank()) {
            return store.list(null);
        }
        TicketStatus parsed = TicketStatus.parse(status);
        if (parsed == null) {
            throw new InvalidTicketStatusException(status);
        }
        return store.list(parsed);
    }

    @PostMapping("/{id}/accept")
    public Ticket accept(@PathVariable long id) {
        return store.transition(id, TicketStatus.IN_PROGRESS);
    }

    @PostMapping("/{id}/resolve")
    public Ticket resolve(@PathVariable long id) {
        return store.transition(id, TicketStatus.RESOLVED);
    }

    @PostMapping("/{id}/close")
    public Ticket close(@PathVariable long id) {
        return store.transition(id, TicketStatus.CLOSED);
    }

    @ExceptionHandler(TicketNotFoundException.class)
    public ResponseEntity<Map<String, Object>> notFound() {
        return error(ErrorCode.TICKET_NOT_FOUND);
    }

    @ExceptionHandler(TicketTransitionException.class)
    public ResponseEntity<Map<String, Object>> invalidTransition() {
        return error(ErrorCode.TICKET_TRANSITION_INVALID);
    }

    @ExceptionHandler(InvalidTicketStatusException.class)
    public ResponseEntity<Map<String, Object>> invalidStatus() {
        return error(ErrorCode.INVALID_REQUEST);
    }

    private static ResponseEntity<Map<String, Object>> error(ErrorCode code) {
        return ResponseEntity.status(code.httpStatus()).body(ErrorEvents.payload(code));
    }
}
