package com.example.csagent.mock.model;

/** GET /api/mock/users 的瘦身条目，供前端用户切换器使用。 */
public record UserSummary(String userId, String name, String tier) {}
