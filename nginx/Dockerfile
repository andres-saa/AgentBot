FROM nginx:1.27-alpine
COPY docker-entrypoint.d/ /docker-entrypoint.d/
RUN chmod +x /docker-entrypoint.d/*.sh || true      && sed -i 's/\r$//' /docker-entrypoint.d/*.sh || true
