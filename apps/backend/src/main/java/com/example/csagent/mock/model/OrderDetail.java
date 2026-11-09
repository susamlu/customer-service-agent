package com.example.csagent.mock.model;

import java.util.List;

/** 聚合后的订单视图：join 商品明细与物流轨迹，日期已换算为绝对值。 */
public record OrderDetail(
        String orderId,
        String status,
        String statusLabel,
        String placedAt,
        int placedDaysAgo,
        double totalAmount,
        String paymentMethod,
        Integer promisedDays,
        boolean presale,
        Integer expectedShipDays,
        String remark,
        List<ItemDetail> items,
        LogisticsDetail logistics) {

    public record ItemDetail(
            String productId,
            String name,
            String category,
            double price,
            int quantity,
            boolean supportsPriceProtection,
            int priceProtectionDays,
            boolean sevenDayReturn,
            String returnCondition) {}

    public record LogisticsDetail(
            String carrier,
            String trackingNo,
            List<TraceDetail> traces) {

        public record TraceDetail(String date, int daysAgo, String location, String desc) {}
    }
}
