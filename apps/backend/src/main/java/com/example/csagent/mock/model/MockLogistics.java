package com.example.csagent.mock.model;

import java.util.List;

/** mock/logistics.json 的原始结构。 */
public record MockLogistics(
        String logisticsId,
        String carrier,
        String trackingNo,
        List<Trace> traces) {

    public record Trace(int daysAgo, String location, String desc) {}
}
