FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU version (sabse pehle)
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Baaki packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
