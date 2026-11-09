package com.example.csagent.obs;

import io.micrometer.core.instrument.config.MeterFilter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 把 SSE 从通用 HTTP 直方图里摘出去。
 *
 * <p>{@code http.server.requests} 记录的是「处理完整个响应」的耗时，而
 * {@code POST /api/chat} 的响应体就是生成流本身：混进去的样本动辄数秒，
 * P95 会被回答长度主导，掩盖真正的网关开销。问答链路的耗时改由
 * {@link ChatMetrics} 的分段计时器负责。
 */
@Configuration
public class HttpMetricsConfig {

    private static final String HTTP_SERVER_REQUESTS = "http.server.requests";
    private static final String CHAT_URI = "/api/chat";

    @Bean
    public MeterFilter dropSseRequestTimer() {
        return MeterFilter.deny(id -> HTTP_SERVER_REQUESTS.equals(id.getName())
                && CHAT_URI.equals(id.getTag("uri")));
    }
}
