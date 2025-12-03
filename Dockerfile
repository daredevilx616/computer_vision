# Python/OpenCV runtime for the CV demos (Flask entrypoint: app.py)
# Works on Render/Railway/Docker hosts with libGL available.

FROM python:3.10-slim

# Install OpenCV runtime deps (libGL + glib)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Default to gunicorn serving the Flask app (app.py)
ENV PORT=8000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}"]
