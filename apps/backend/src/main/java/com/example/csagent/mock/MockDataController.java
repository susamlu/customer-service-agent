package com.example.csagent.mock;

import com.example.csagent.mock.model.UserContext;
import com.example.csagent.mock.model.UserSummary;
import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/** mock 业务数据接口：模拟订单/商品/物流/用户系统，供模型服务与前端调用。 */
@RestController
@RequestMapping("/api/mock")
public class MockDataController {

    private final MockDataService data;

    public MockDataController(MockDataService data) {
        this.data = data;
    }

    /** 用户列表，前端切换器使用。 */
    @GetMapping("/users")
    public List<UserSummary> users() {
        return data.users();
    }

    /** 一个用户的完整业务上下文：用户信息 + 订单（内嵌商品明细与物流轨迹）。 */
    @GetMapping("/context")
    public ResponseEntity<UserContext> context(@RequestParam String userId) {
        return data.context(userId)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }
}
