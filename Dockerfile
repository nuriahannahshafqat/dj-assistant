FROM python:3.12-slim

# ffmpeg decodes the AAC audio previews for BPM/key analysis.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Railway sets $PORT. Audio analysis takes a few seconds, so allow a long timeout.
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120
