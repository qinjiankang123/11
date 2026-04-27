FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 直接用固定端口 8080，和代码里的默认端口保持一致
CMD uvicorn app:app --host 0.0.0.0 --port 8080