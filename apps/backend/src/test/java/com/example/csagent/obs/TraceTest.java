package com.example.csagent.obs;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** trace_id 语义（§11）：合法值继承，脏值替换，绝不因为一个头让请求失败。 */
class TraceTest {

    @Test
    @DisplayName("valid incoming id is inherited verbatim")
    void validIncomingIsInherited() {
        assertThat(Trace.resolve("abc_123-xyz.9")).isEqualTo("abc_123-xyz.9");
    }

    @Test
    @DisplayName("missing id gets a fresh 16-char id")
    void missingGetsGenerated() {
        String id = Trace.resolve(null);
        assertThat(id).hasSize(16);
        assertThat(id).matches("[0-9a-f]{16}");
    }

    @Test
    @DisplayName("illegal characters, empty and overlong ids are replaced, not rejected")
    void invalidIsReplaced() {
        assertThat(Trace.resolve("bad value")).hasSize(16).isNotEqualTo("bad value");
        assertThat(Trace.resolve("")).hasSize(16);
        assertThat(Trace.resolve("a".repeat(65))).hasSize(16);
        assertThat(Trace.resolve("a".repeat(64))).hasSize(64);
    }

    @Test
    @DisplayName("generated ids do not collide")
    void generatedIdsDiffer() {
        assertThat(Trace.newId()).isNotEqualTo(Trace.newId());
    }
}
