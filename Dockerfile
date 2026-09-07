FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements-local-dev.txt /app/requirements-local-dev.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements-local-dev.txt && \
    groupadd --gid 10001 craftly && \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin craftly

COPY --chown=10001:10001 . /app
USER 10001:10001
EXPOSE 8090
CMD ["python", "-m", "uvicorn", "src.craftly.gateway.api:app", "--host", "0.0.0.0", "--port", "8090"]

