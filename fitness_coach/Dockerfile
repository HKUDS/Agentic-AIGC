# Image for the Telegram fitness bot. Build context is the repository root:
#   docker build -f fitness_coach/Dockerfile -t fitness-coach .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FITNESS_DB_PATH=/data/bot.sqlite3

WORKDIR /app

# The bot needs nothing but aiohttp and the standard library, so the image
# stays small and the build stays fast.
RUN pip install --no-cache-dir "aiohttp>=3.12.14"

COPY fitness_coach /app/fitness_coach

# Runs as root on purpose: Railway mounts persistent volumes root-owned, and a
# non-root user could not write the SQLite file there.
RUN mkdir -p /data

# The web sign-up page listens here; PaaS platforms override it through PORT.
ENV FITNESS_WEB_PORT=8080
EXPOSE 8080

CMD ["python", "-m", "fitness_coach"]
