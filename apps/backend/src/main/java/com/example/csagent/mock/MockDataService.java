package com.example.csagent.mock;

import com.example.csagent.mock.model.MockLogistics;
import com.example.csagent.mock.model.MockOrder;
import com.example.csagent.mock.model.MockProduct;
import com.example.csagent.mock.model.MockUser;
import com.example.csagent.mock.model.OrderDetail;
import com.example.csagent.mock.model.UserContext;
import com.example.csagent.mock.model.UserSummary;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

/**
 * 启动时加载 resources/mock/ 下四份 JSON，聚合为用户视角的业务上下文。
 *
 * <p>数据里的日期统一用 daysAgo（相对天数）表达，加载聚合时换算为绝对日期，
 * 保证 demo 无论何时运行，「物流停滞 2 天」「价保刚过期」等剧本始终新鲜。
 */
@Component
public class MockDataService {

    private final ObjectMapper mapper = new ObjectMapper();

    private List<MockUser> users = List.of();
    private Map<String, MockOrder> ordersById = Map.of();
    private Map<String, MockProduct> productsById = Map.of();
    private Map<String, MockLogistics> logisticsById = Map.of();

    @PostConstruct
    void load() {
        try {
            users = List.of(read("mock/users.json", MockUser[].class));
            ordersById = List.of(read("mock/orders.json", MockOrder[].class)).stream()
                    .collect(Collectors.toMap(MockOrder::orderId, Function.identity()));
            productsById = List.of(read("mock/products.json", MockProduct[].class)).stream()
                    .collect(Collectors.toMap(MockProduct::productId, Function.identity()));
            logisticsById = List.of(read("mock/logistics.json", MockLogistics[].class)).stream()
                    .collect(Collectors.toMap(MockLogistics::logisticsId, Function.identity()));
        } catch (IOException e) {
            throw new UncheckedIOException("mock 数据加载失败", e);
        }
    }

    private <T> T read(String path, Class<T> type) throws IOException {
        try (var in = new ClassPathResource(path).getInputStream()) {
            return mapper.readValue(in, type);
        }
    }

    /** 用户列表（瘦身版），供前端切换器。 */
    public List<UserSummary> users() {
        return users.stream()
                .map(u -> new UserSummary(u.userId(), u.name(), u.tier()))
                .toList();
    }

    /** 聚合一个用户的全部业务上下文；用户不存在返回 empty。 */
    public Optional<UserContext> context(String userId) {
        return users.stream()
                .filter(u -> u.userId().equals(userId))
                .findFirst()
                .map(u -> new UserContext(
                        u.userId(),
                        u.name(),
                        u.nickname(),
                        u.tier(),
                        dateAgo(u.registeredDaysAgo()),
                        u.orderIds().stream()
                                .map(ordersById::get)
                                .filter(o -> o != null)
                                .map(this::toDetail)
                                .toList()));
    }

    private OrderDetail toDetail(MockOrder order) {
        List<OrderDetail.ItemDetail> items = order.items().stream()
                .map(line -> {
                    MockProduct p = productsById.get(line.productId());
                    if (p == null) {
                        throw new IllegalStateException(
                                "mock 数据引用了不存在的商品: " + line.productId());
                    }
                    return new OrderDetail.ItemDetail(
                            p.productId(), p.name(), p.category(), p.price(), line.quantity(),
                            p.supportsPriceProtection(), p.priceProtectionDays(),
                            p.sevenDayReturn(), p.returnCondition());
                })
                .toList();

        MockLogistics logistics = order.logisticsId() == null
                ? null
                : logisticsById.get(order.logisticsId());
        OrderDetail.LogisticsDetail logisticsDetail = logistics == null ? null
                : new OrderDetail.LogisticsDetail(
                        logistics.carrier(),
                        logistics.trackingNo(),
                        logistics.traces().stream()
                                .map(t -> new OrderDetail.LogisticsDetail.TraceDetail(
                                        dateAgo(t.daysAgo()), t.daysAgo(), t.location(), t.desc()))
                                .toList());

        return new OrderDetail(
                order.orderId(),
                order.status(),
                order.statusLabel(),
                dateAgo(order.placedDaysAgo()),
                order.placedDaysAgo(),
                order.totalAmount(),
                order.paymentMethod(),
                order.promisedDays(),
                order.presale(),
                order.expectedShipDays(),
                order.remark(),
                items,
                logisticsDetail);
    }

    private String dateAgo(int daysAgo) {
        return LocalDate.now().minusDays(daysAgo).toString();
    }
}
