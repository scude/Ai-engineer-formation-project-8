# syntax=docker/dockerfile:1
FROM python:3.10-slim

# ---- Environnement ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/app:${PYTHONPATH}" \
    PORT=8000

WORKDIR /app

# ---- Dépendances ----
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Code applicatif ----
COPY app ./app
COPY notebook/scripts ./notebook/scripts

# ---- Lancement ----
CMD ["sh", "-c", "gunicorn app.wsgi:app --workers=1 --threads=4 --timeout=120 --bind 0.0.0.0:$PORT"]
