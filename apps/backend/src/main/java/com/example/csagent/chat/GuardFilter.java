package com.example.csagent.chat;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * 入口护栏（architecture.md §5.1）：认证 → 限流 → 长度上限。
 *
 * <p>只拦 {@code POST /api/chat}：只有这个接口会触发检索与生成，历史读取没有成本。
 * 拒绝时按 §3.5 的统一封装回 JSON，直接结束，不进入编排器。
 *
 * <p>限流是<b>按会话</b>的内存令牌桶，进程重启即清零（与内存 SessionStore 同一取舍）。
 * sessionId 数量无上限，所以桶表设了容量上限并定期清扫，防止被随机 ID 撑爆。
 */
@Component
public class GuardFilter extends OncePerRequestFilter {

    private static final Logger log = LoggerFactory.getLogger(GuardFilter.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String CHAT_PATH = "/api/chat";
    private static final String TOKEN_HEADER = "X-Api-Token";
    private static final int MAX_BUCKETS = 10_000;
    private static final long BUCKET_IDLE_MILLIS = TimeUnit.MINUTES.toMillis(10);

    private final String apiToken;
    private final int rateLimitPerMinute;
    private final long maxBodyBytes;
    private final int maxSessionIdLength;

    private final Map<String, Bucket> buckets = new ConcurrentHashMap<>();

    public GuardFilter(
            @Value("${guard.api-token:}") String apiToken,
            @Value("${guard.rate-limit-per-minute:20}") int rateLimitPerMinute,
            @Value("${guard.max-body-bytes:4096}") long maxBodyBytes,
            @Value("${guard.max-session-id-length:64}") int maxSessionIdLength) {
        this.apiToken = apiToken == null ? "" : apiToken.trim();
        this.rateLimitPerMinute = rateLimitPerMinute;
        this.maxBodyBytes = maxBodyBytes;
        this.maxSessionIdLength = maxSessionIdLength;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        return !CHAT_PATH.equals(request.getRequestURI())
                || !"POST".equalsIgnoreCase(request.getMethod());
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        if (!authorized(request)) {
            reject(response, ErrorCode.UNAUTHORIZED);
            return;
        }
        if (request.getContentLengthLong() > maxBodyBytes) {
            reject(response, ErrorCode.REQUEST_TOO_LARGE);
            return;
        }
        String sessionId = request.getParameter("sessionId");
        if (sessionId == null
                || sessionId.isBlank()
                || sessionId.length() > maxSessionIdLength) {
            reject(response, ErrorCode.INVALID_REQUEST);
            return;
        }
        long now = System.currentTimeMillis();
        if (!allow(sessionId, now)) {
            log.warn("rate limited session={}", sessionId);
            reject(response, ErrorCode.RATE_LIMITED);
            return;
        }
        chain.doFilter(request, response);
    }

    /** 未配置 token 时认证关闭（本地/演示默认）；配置后必须带一致的头。 */
    private boolean authorized(HttpServletRequest request) {
        if (apiToken.isEmpty()) {
            return true;
        }
        String provided = request.getHeader(TOKEN_HEADER);
        if (provided == null) {
            return false;
        }
        // 定长比较，避免按字节短路泄漏 token 前缀。
        return MessageDigest.isEqual(
                provided.getBytes(StandardCharsets.UTF_8),
                apiToken.getBytes(StandardCharsets.UTF_8));
    }

    /** 令牌桶：容量 = 每分钟额度，按时间线性回填。 */
    private boolean allow(String sessionId, long now) {
        if (buckets.size() > MAX_BUCKETS) {
            buckets.entrySet().removeIf(
                    entry -> now - entry.getValue().updatedMillis() > BUCKET_IDLE_MILLIS);
        }
        Bucket updated = buckets.compute(sessionId, (key, bucket) -> {
            if (bucket == null) {
                return new Bucket(rateLimitPerMinute - 1.0, now);
            }
            double refill = (now - bucket.updatedMillis())
                    / (double) TimeUnit.MINUTES.toMillis(1)
                    * rateLimitPerMinute;
            return new Bucket(
                    Math.min(rateLimitPerMinute, bucket.tokens() + refill) - 1.0, now);
        });
        return updated.tokens() >= 0;
    }

    private void reject(HttpServletResponse response, ErrorCode code) throws IOException {
        response.setStatus(code.httpStatus());
        response.setContentType("application/json;charset=UTF-8");
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("code", code.name());
        body.put("message", code.userMessage());
        body.put("retriable", code.retriable());
        response.getWriter().write(MAPPER.writeValueAsString(body));
    }

    private record Bucket(double tokens, long updatedMillis) {}
}
