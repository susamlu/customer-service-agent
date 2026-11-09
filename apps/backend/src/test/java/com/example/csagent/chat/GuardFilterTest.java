package com.example.csagent.chat;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

/** 入口护栏（§5.1）：认证 → 请求体 → sessionId → 限流。 */
class GuardFilterTest {

    private static final String TOKEN = "secret-token";

    private static GuardFilter filter() {
        return new GuardFilter("", 20, 4096, 64);
    }

    private static MockHttpServletRequest chat(String sessionId) {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/chat");
        if (sessionId != null) {
            request.setParameter("sessionId", sessionId);
        }
        return request;
    }

    private static MockFilterChain pass(GuardFilter filter, MockHttpServletRequest request)
            throws Exception {
        MockFilterChain chain = new MockFilterChain();
        filter.doFilter(request, new MockHttpServletResponse(), chain);
        return chain;
    }

    @Test
    @DisplayName("only POST /api/chat is guarded")
    void shouldNotFilter_otherPathsAndMethods() {
        GuardFilter filter = filter();

        assertThat(filter.shouldNotFilter(new MockHttpServletRequest("GET", "/api/chat"))).isTrue();
        assertThat(filter.shouldNotFilter(new MockHttpServletRequest("POST", "/api/history")))
                .isTrue();
        assertThat(filter.shouldNotFilter(chat("s1"))).isFalse();
    }

    @Test
    @DisplayName("blank token disables auth")
    void blankToken_allowsRequest() throws Exception {
        MockFilterChain chain = pass(filter(), chat("s1"));
        assertThat(chain.getRequest()).isNotNull();
    }

    @Test
    @DisplayName("configured token requires an exact header")
    void configuredToken_missingOrWrong_returns401() throws Exception {
        GuardFilter filter = new GuardFilter(TOKEN, 20, 4096, 64);

        MockHttpServletResponse missing = new MockHttpServletResponse();
        filter.doFilter(chat("s1"), missing, new MockFilterChain());
        assertThat(missing.getStatus()).isEqualTo(401);
        assertThat(missing.getContentAsString()).contains("UNAUTHORIZED");

        MockHttpServletRequest wrong = chat("s1");
        wrong.addHeader("X-Api-Token", "nope");
        MockHttpServletResponse wrongResponse = new MockHttpServletResponse();
        filter.doFilter(wrong, wrongResponse, new MockFilterChain());
        assertThat(wrongResponse.getStatus()).isEqualTo(401);

        MockHttpServletRequest ok = chat("s1");
        ok.addHeader("X-Api-Token", TOKEN);
        MockFilterChain chain = pass(filter, ok);
        assertThat(chain.getRequest()).isNotNull();
    }

    @Test
    @DisplayName("body over the limit is rejected with 413")
    void oversizedBody_returns413() throws Exception {
        MockHttpServletRequest request = chat("s1");
        request.setContent(new byte[4097]);

        MockHttpServletResponse response = new MockHttpServletResponse();
        filter().doFilter(request, response, new MockFilterChain());

        assertThat(response.getStatus()).isEqualTo(413);
        assertThat(response.getContentAsString()).contains("REQUEST_TOO_LARGE");
    }

    @Test
    @DisplayName("missing, blank or oversized sessionId is rejected with 400")
    void invalidSessionId_returns400() throws Exception {
        for (MockHttpServletRequest request : new MockHttpServletRequest[] {
                chat(null), chat("   "), chat("x".repeat(65))}) {
            MockHttpServletResponse response = new MockHttpServletResponse();
            filter().doFilter(request, response, new MockFilterChain());

            assertThat(response.getStatus()).isEqualTo(400);
            assertThat(response.getContentAsString()).contains("INVALID_REQUEST");
        }
    }

    @Test
    @DisplayName("the 21st request in a minute from one session is rate limited")
    void rateLimit_blocksAfterQuota() throws Exception {
        GuardFilter filter = filter();
        for (int i = 0; i < 20; i++) {
            assertThat(pass(filter, chat("s1")).getRequest()).isNotNull();
        }

        MockHttpServletResponse response = new MockHttpServletResponse();
        filter.doFilter(chat("s1"), response, new MockFilterChain());

        assertThat(response.getStatus()).isEqualTo(429);
        assertThat(response.getContentAsString())
                .contains("RATE_LIMITED")
                .contains("\"retriable\":true");
    }

    @Test
    @DisplayName("rate limit buckets are per session")
    void rateLimit_isPerSession() throws Exception {
        GuardFilter filter = new GuardFilter("", 1, 4096, 64);

        assertThat(pass(filter, chat("a")).getRequest()).isNotNull();
        assertThat(pass(filter, chat("b")).getRequest()).isNotNull();

        MockHttpServletResponse second = new MockHttpServletResponse();
        filter.doFilter(chat("a"), second, new MockFilterChain());
        assertThat(second.getStatus()).isEqualTo(429);
    }
}
