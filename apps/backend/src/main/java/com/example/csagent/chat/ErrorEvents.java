package com.example.csagent.chat;

import com.example.csagent.obs.Trace;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 错误信封（architecture.md §3.5）：SSE 错误事件与 REST 错误体同形，
 * 前端一套解析逻辑通吃。编排器、控制器、工单接口共用这一处构造，避免三份漂移。
 */
public final class ErrorEvents {

    private ErrorEvents() {}

    public static Map<String, Object> payload(ErrorCode code) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("type", "error");
        payload.put("code", code.name());
        payload.put("message", code.userMessage());
        payload.put("retriable", code.retriable());
        String traceId = Trace.current();
        if (traceId != null) {
            payload.put("traceId", traceId);
        }
        return payload;
    }
}
