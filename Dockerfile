FROM python:3.9-slim

WORKDIR /app

COPY src/api_server.py /app/api_server.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "api_server.py"]
