FROM python:3.12.15-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3
RUN apt-get update && apt-get install -y --no-install-recommends openbabel autodock-vina \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m venv /app/.venv && /app/.venv/bin/python -m pip install --no-cache-dir -r /app/requirements.txt
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /mnt
CMD ["python", "--version"]
