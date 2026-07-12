
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM builder AS tester

COPY . .

RUN python3 -m py_compile *.py

RUN touch /app/.tests_passed


FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap0.8 \
    tcpdump \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=tester /app/.tests_passed /app/.tests_passed

COPY . .

ENV NIDS_MODEL_DIR=/app/models
ENV NIDS_LOG_DIR=/app/logs
ENV NIDS_LOG_LEVEL=INFO

RUN mkdir -p /app/logs /app/models

CMD ["python3", "main.py"]