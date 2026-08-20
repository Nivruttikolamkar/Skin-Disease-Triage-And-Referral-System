FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and C extensions
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["sh", "-c", "python3 migrate_db.py && gunicorn --bind 0.0.0.0:5000 --timeout 120 app:app"]
