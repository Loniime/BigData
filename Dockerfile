FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ✅ Installer Docker CLI + dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
RUN pip install --no-cache-dir uv

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock* ./

# Créer le venv et installer les dépendances
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e .

# Copier le code source
COPY src ./src
COPY main.py ./main.py
COPY README.md* ./

# Activer le venv par défaut
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]