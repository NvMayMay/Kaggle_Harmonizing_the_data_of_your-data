FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build databases if not present
RUN python build_databases.py || true

EXPOSE 8080

CMD ["python", "server.py", "--transport", "http", "--port", "8080"]
