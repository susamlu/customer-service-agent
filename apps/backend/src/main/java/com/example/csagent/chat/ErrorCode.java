package com.example.csagent.chat;

/**
 * 对外错误码（architecture.md §9）。
 *
 * <p>枚举里只放可以给用户看的话术。底层异常文本、端口号、堆栈一律只进日志：
 * 铁律是任何异常都不允许把 getMessage() 直接塞进响应。
 */
public enum ErrorCode {
    INVALID_REQUEST(400, false, "请求格式不正确。"),
    UNAUTHORIZED(401, false, "身份校验失败，请刷新页面后重试。"),
    REQUEST_TOO_LARGE(413, false, "问题描述过长，请精简后再试。"),
    RATE_LIMITED(429, true, "请求太频繁，请稍后再试。"),
    ROUTE_LOW_CONFIDENCE(400, false, "能否具体描述一下您的问题？"),
    ORDER_NOT_FOUND(404, false, "没有找到该订单，请核对订单号后重试。"),
    ORDER_CONTEXT_MISSING(400, false, "请提供订单号，我来帮您查询。"),
    RETRIEVAL_EMPTY(200, false, "知识库暂无匹配，以下回答供参考。"),
    // 工单闭环（§6）：不存在与非法状态迁移是两种客户端错误，不合并。
    TICKET_NOT_FOUND(404, false, "工单不存在。"),
    TICKET_TRANSITION_INVALID(409, false, "工单当前状态不支持该操作。"),
    // 可答性闸门判定 no_context：知识库没有可信依据，不调生成（§5.4）。
    NO_GROUNDED_CONTEXT(200, false, AnswerabilityGate.NO_CONTEXT_REPLY),
    UPSTREAM_TIMEOUT(504, true, "上游服务超时，请稍后重试。"),
    UPSTREAM_RATE_LIMITED(429, true, "服务繁忙，请稍后重试。"),
    STREAM_INTERRUPTED(200, true, "回答中断，请重试。"),
    INTERNAL(500, true, "服务暂时不可用，请稍后重试。");

    private final int httpStatus;
    private final boolean retriable;
    private final String userMessage;

    ErrorCode(int httpStatus, boolean retriable, String userMessage) {
        this.httpStatus = httpStatus;
        this.retriable = retriable;
        this.userMessage = userMessage;
    }

    public int httpStatus() {
        return httpStatus;
    }

    public boolean retriable() {
        return retriable;
    }

    public String userMessage() {
        return userMessage;
    }

    /** 未知或缺失的 code 一律折成 INTERNAL，不做猜测。 */
    public static ErrorCode fromCode(String code) {
        if (code == null) {
            return INTERNAL;
        }
        for (ErrorCode candidate : values()) {
            if (candidate.name().equals(code)) {
                return candidate;
            }
        }
        return INTERNAL;
    }
}
