FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker layer caching works well.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Non-root user for a bit of defense in depth.
RUN useradd --create-home --uid 1000 reviewer && chown -R reviewer /app
USER reviewer

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
