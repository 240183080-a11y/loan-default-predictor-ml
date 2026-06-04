# 1. Use an official lightweight Python base image
FROM python:3.11-slim

# 2. Install uv directly inside the container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set working directory inside the container
WORKDIR /app

# 4. Copy dependency file first (for Docker caching optimizations)
COPY requirements.txt .

# 5. Fast-install dependencies using uv
RUN uv pip install --system -r requirements.txt

# 6. Copy the rest of your project code
COPY . .

# 🔑 CRITICAL FEATURE ALIGNMENT MAPPING:
RUN mkdir -p /app/model
COPY src/serving/model/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/preprocessing.pkl /app/model/preprocessing.pkl

ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]