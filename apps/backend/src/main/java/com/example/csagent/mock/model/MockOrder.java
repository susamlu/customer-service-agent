package com.example.csagent.mock.model;

import java.util.List;

/** mock/orders.json 的原始结构。 */
public record MockOrder(
        String orderId,
        String userId,
        String status,
        String statusLabel,
        int placedDaysAgo,
        double totalAmount,
        String paymentMethod,
        List<LineItem> items,
        String logisticsId,
        Integer promisedDays,
        boolean presale,
        Integer expectedShipDays,
        String remark) {

    public record LineItem(String productId, int quantity) {}
}
