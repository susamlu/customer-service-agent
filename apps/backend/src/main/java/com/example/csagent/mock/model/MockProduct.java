package com.example.csagent.mock.model;

/** mock/products.json 的原始结构。 */
public record MockProduct(
        String productId,
        String name,
        String category,
        double price,
        boolean supportsPriceProtection,
        int priceProtectionDays,
        boolean sevenDayReturn,
        String returnCondition) {}
