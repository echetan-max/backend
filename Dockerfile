# Install Flask and expose dummy server
RUN pip install flask
COPY dummy_server.py /app/
CMD ["python", "dummy_server.py"]
