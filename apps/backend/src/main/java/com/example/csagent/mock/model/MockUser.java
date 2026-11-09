package com.example.csagent.mock.model;

import java.util.List;

/** mock/users.json 的原始结构。 */
public record MockUser(
        String userId,
        String name,
        String nickname,
        String tier,
        int registeredDaysAgo,
        List<String> orderIds) {}
