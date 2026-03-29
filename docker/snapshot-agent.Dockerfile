FROM python:3.12-slim

RUN pip install --no-cache-dir httpx

COPY python/podstack-worker/src/podstack_worker/snapshot_agent.py /app/snapshot_agent.py
COPY docker/snapshot-agent-entrypoint.py /app/entrypoint.py

WORKDIR /app

ENTRYPOINT ["python", "-u", "entrypoint.py"]
