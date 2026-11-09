package com.example.csagent.obs;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import org.slf4j.MDC;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * 请求入口生成/继承 trace_id，并写进 MDC 与响应头（architecture.md §11）。
 *
 * <p>排在最前：鉴权拒绝、限流拒绝也要有 trace_id，否则线上排障时最先被拒的那批请求
 * 恰恰没有线索。异步分派不再重复执行（{@code OncePerRequestFilter} 默认跳过 ASYNC），
 * 流式线程的 MDC 由 {@code ChatController} 在 lambda 内重新挂载。
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class TraceFilter extends OncePerRequestFilter {

    @Override
    protected void doFilterInternal(
            HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        String traceId = Trace.resolve(request.getHeader(Trace.HEADER));
        MDC.put(Trace.MDC_KEY, traceId);
        response.setHeader(Trace.HEADER, traceId);
        try {
            chain.doFilter(request, response);
        } finally {
            MDC.remove(Trace.MDC_KEY);
        }
    }
}
