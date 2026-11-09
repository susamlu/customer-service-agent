# 网关容器：node 构建前端 → maven 打 jar → jre 运行。
# 构建上下文为项目根（见同目录 docker-compose.yml）。
FROM node:20-slim AS frontend
WORKDIR /build
COPY apps/frontend/package.json apps/frontend/package-lock.json* ./frontend/
RUN cd frontend && npm install --silent
COPY apps/frontend/ frontend/
# vite outDir 指向 ../backend/src/main/resources/static，即 /build/backend/...
RUN cd frontend && npm run build

FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /build
COPY apps/backend/pom.xml backend/pom.xml
RUN mvn -f backend/pom.xml dependency:go-offline -q
COPY apps/backend/src backend/src
COPY --from=frontend /build/backend/src/main/resources/static \
     backend/src/main/resources/static
RUN mvn -f backend/pom.xml package -DskipTests -q

FROM eclipse-temurin:17-jre
WORKDIR /app
COPY --from=build /build/backend/target/csagent-backend-1.0.0.jar app.jar
EXPOSE 8090
ENTRYPOINT ["java", "-jar", "app.jar"]
