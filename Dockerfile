# Use official Python slim image
FROM python:3.11-slim

# Install system deps (optional but good for numpy/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install Flask (for dummy server)
RUN pip install flask

# Copy backend code
COPY worker.py sdc_io.py pipeline_encode.py pipeline_decode.py dummy_server.py /app/

# Expose dummy port for Render
EXPOSE 10000

# Start dummy server (runs worker loop in background)
CMD ["python", "dummy_server.py"]
