package com.example.csagent.obs;

import java.util.UUID;
import java.util.regex.Pattern;
import org.slf4j.MDC;

/**
 * trace_id 的唯一约定（architecture.md §11）：一个请求一个 id，日志、SSE 与上游调用共用。
 *
 * <p>ID 只允许 {@code [A-Za-z0-9._-]{1,64}}：来自请求头的值会被写进日志与响应头，
 * 放行换行或引号就可能被日志注入；非法值不报错，直接换新 ID，追踪不该因为脏头而失败。
 */
public final class Trace {

    public static final String HEADER = "X-Trace-Id";
    public static final String MDC_KEY = "traceId";

    private static final Pattern VALID = Pattern.compile("[A-Za-z0-9._-]{1,64}");

    private Trace() {}

    /** 当前线程的 trace_id；不在请求上下文中时为 null。 */
    public static String current() {
        return MDC.get(MDC_KEY);
    }

    /** 合法就沿用上游 ID（跨网关/模型服务串联），否则生成新 ID。 */
    public static String resolve(String incoming) {
        return incoming != null && VALID.matcher(incoming).matches() ? incoming : newId();
    }

    public static String newId() {
        return UUID.randomUUID().toString().replace("-", "").substring(0, 16);
    }
}
