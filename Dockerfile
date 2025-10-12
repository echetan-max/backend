FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask

COPY worker.py sdc_io.py pipeline_encode.py pipeline_decode.py dummy_server.py /app/

EXPOSE 10000

CMD ["python", "dummy_server.py"]
