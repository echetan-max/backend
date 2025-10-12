FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY worker.py sdc_io.py pipeline_encode.py pipeline_decode.py /app/

ENV PYTHONUNBUFFERED=1
CMD ["python", "worker.py"]
