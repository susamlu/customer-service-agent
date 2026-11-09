package com.example.csagent.ticket;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

/** 工单接口（§6）：正常读取/流转，以及 404 / 409 / 400 三条错误路径。 */
class TicketControllerTest {

    private final TicketStore store = mock(TicketStore.class);
    private MockMvc mvc;

    @BeforeEach
    void setUp() {
        mvc = MockMvcBuilders.standaloneSetup(new TicketController(store)).build();
    }

    private static Ticket ticket(long id, TicketStatus status) {
        return new Ticket(id, "s1", "u1", status, "q", null, 1L, 2L);
    }

    @Test
    @DisplayName("GET returns an existing ticket")
    void getReturnsTicket() throws Exception {
        when(store.find(1)).thenReturn(Optional.of(ticket(1, TicketStatus.PENDING)));

        mvc.perform(get("/api/tickets/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.status").value("PENDING"));
    }

    @Test
    @DisplayName("GET on a missing ticket returns 404 with the shared error envelope")
    void getMissingReturns404() throws Exception {
        when(store.find(99)).thenReturn(Optional.empty());

        mvc.perform(get("/api/tickets/99"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.type").value("error"))
                .andExpect(jsonPath("$.code").value("TICKET_NOT_FOUND"));
    }

    @Test
    @DisplayName("accept delegates the pending -> in progress transition")
    void acceptDelegatesTransition() throws Exception {
        when(store.transition(1, TicketStatus.IN_PROGRESS))
                .thenReturn(ticket(1, TicketStatus.IN_PROGRESS));

        mvc.perform(post("/api/tickets/1/accept"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("IN_PROGRESS"));

        verify(store).transition(1, TicketStatus.IN_PROGRESS);
    }

    @Test
    @DisplayName("resolve delegates the in progress -> resolved transition")
    void resolveDelegatesTransition() throws Exception {
        when(store.transition(1, TicketStatus.RESOLVED))
                .thenReturn(ticket(1, TicketStatus.RESOLVED));

        mvc.perform(post("/api/tickets/1/resolve"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("RESOLVED"));
    }

    @Test
    @DisplayName("an illegal transition surfaces as 409, never as a silent overwrite")
    void illegalTransitionReturns409() throws Exception {
        when(store.transition(1, TicketStatus.RESOLVED))
                .thenThrow(new TicketTransitionException(1, TicketStatus.CLOSED, TicketStatus.RESOLVED));

        mvc.perform(post("/api/tickets/1/resolve"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code").value("TICKET_TRANSITION_INVALID"));
    }

    @Test
    @DisplayName("list filters by a recognised status")
    void listFiltersByStatus() throws Exception {
        when(store.list(TicketStatus.PENDING))
                .thenReturn(List.of(ticket(1, TicketStatus.PENDING)));

        mvc.perform(get("/api/tickets").param("status", "pending"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].status").value("PENDING"));
    }

    @Test
    @DisplayName("list without a status returns everything")
    void listWithoutStatusReturnsAll() throws Exception {
        when(store.list(null)).thenReturn(List.of());

        mvc.perform(get("/api/tickets"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray());
    }

    @Test
    @DisplayName("an unknown status filter is a 400, not an empty list")
    void unknownStatusReturns400() throws Exception {
        mvc.perform(get("/api/tickets").param("status", "archived"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("INVALID_REQUEST"));

        verifyNoInteractions(store);
    }
}
