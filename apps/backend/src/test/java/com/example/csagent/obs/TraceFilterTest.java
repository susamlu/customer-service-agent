package com.example.csagent.obs;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import jakarta.servlet.ServletException;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.MDC;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

/** 入口过滤器：响应头回写 trace_id、MDC 在链内可见、出链必清（§11）。 */
class TraceFilterTest {

    private final TraceFilter filter = new TraceFilter();

    @AfterEach
    void clearMdc() {
        MDC.clear();
    }

    @Test
    @DisplayName("valid inbound trace id is echoed and visible to downstream code")
    void validIdEchoedAndVisible() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader(Trace.HEADER, "trace-abc");
        MockHttpServletResponse response = new MockHttpServletResponse();
        AtomicReference<String> seen = new AtomicReference<>();

        filter.doFilterInternal(request, response, (req, res) -> seen.set(MDC.get(Trace.MDC_KEY)));

        assertThat(seen.get()).isEqualTo("trace-abc");
        assertThat(response.getHeader(Trace.HEADER)).isEqualTo("trace-abc");
        assertThat(MDC.get(Trace.MDC_KEY)).isNull();
    }

    @Test
    @DisplayName("missing trace id is generated and echoed")
    void missingIdGenerated() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest();
        MockHttpServletResponse response = new MockHttpServletResponse();
        AtomicReference<String> seen = new AtomicReference<>();

        filter.doFilterInternal(request, response, (req, res) -> seen.set(MDC.get(Trace.MDC_KEY)));

        assertThat(seen.get()).hasSize(16);
        assertThat(seen.get()).isEqualTo(response.getHeader(Trace.HEADER));
    }

    @Test
    @DisplayName("illegal inbound trace id is replaced before it can reach logs")
    void illegalIdReplaced() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader(Trace.HEADER, "not a valid value");
        MockHttpServletResponse response = new MockHttpServletResponse();
        AtomicReference<String> seen = new AtomicReference<>();

        filter.doFilterInternal(request, response, (req, res) -> seen.set(MDC.get(Trace.MDC_KEY)));

        assertThat(seen.get()).hasSize(16).isNotEqualTo("not a valid value");
        assertThat(response.getHeader(Trace.HEADER)).isEqualTo(seen.get());
    }

    @Test
    @DisplayName("MDC is cleared even when downstream throws")
    void mdcClearedOnFailure() {
        MockHttpServletRequest request = new MockHttpServletRequest();
        MockHttpServletResponse response = new MockHttpServletResponse();

        assertThatThrownBy(() -> filter.doFilterInternal(
                        request,
                        response,
                        (req, res) -> {
                            throw new ServletException("boom");
                        }))
                .isInstanceOf(ServletException.class);

        assertThat(MDC.get(Trace.MDC_KEY)).isNull();
    }
}
