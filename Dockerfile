# Runs the Gradio dashboard only. Ollama stays on the host and Postgres stays on Supabase,
# so this image carries no models and no database.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Dependencies first so edits to src/ do not invalidate the install layer
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-interaction --no-root

COPY src/ ./src/
COPY app.py ./

EXPOSE 7860

# 0.0.0.0 so the port is reachable from outside the container
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
