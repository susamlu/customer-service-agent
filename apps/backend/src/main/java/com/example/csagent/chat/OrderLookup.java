package com.example.csagent.chat;

import com.example.csagent.mock.model.OrderDetail;

/**
 * 订单工具的一次查询结果。超时与异常不在这里表达，由 {@link FanOut.Outcome} 承载；
 * 这里只区分业务上必须区别对待的三种结局。
 */
public sealed interface OrderLookup {

    /** 命中订单（用户没指定订单号时取最近一笔）。 */
    record Found(OrderDetail order) implements OrderLookup {}

    /** 用户明确给了订单号但查不到：必须如实告知，禁止回退到别的订单（architecture.md §8）。 */
    record Missing(String orderId) implements OrderLookup {}

    /** 当前用户在系统里没有任何订单。 */
    record NoContext() implements OrderLookup {}
}
