package com.example.csagent.chat;

/**
 * 模型服务调用失败的载体。
 *
 * <p>code 决定用户看到什么，message 只用于日志排查，不会进入响应体。
 */
public class ModelServiceException extends RuntimeException {

    private final ErrorCode code;

    public ModelServiceException(ErrorCode code, String detail, Throwable cause) {
        super(detail, cause);
        this.code = code;
    }

    public ErrorCode code() {
        return code;
    }
}
