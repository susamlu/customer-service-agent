package com.example.csagent.mock.model;

import java.util.List;

/** GET /api/mock/context 的聚合响应：一个用户的全部业务上下文。 */
public record UserContext(
        String userId,
        String name,
        String nickname,
        String tier,
        String registeredAt,
        List<OrderDetail> orders) {}
