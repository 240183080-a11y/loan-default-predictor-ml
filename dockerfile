# 1. Use an official lightweight Python base image
FROM python:3.11-slim

# 2. Install uv directly inside the container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set working directory inside the container
WORKDIR /app

# 💡 ADD THIS BLOCK: Install libgomp1 (OpenMP) so LightGBM can run its C++ engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency file first (for Docker caching optimizations)
COPY requirements.txt .

# 5. Fast-install dependencies using uv
RUN uv pip install --system -r requirements.txt

# 6. Copy the rest of your project code (Includes src/, configuration files, and your model/ folder)
COPY . .

# 🔑 CRITICAL FEATURE ALIGNMENT MAPPING:
COPY src/serving/model/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/preprocessing.pkl /app/model/preprocessing.pkl

# 7. Set environment variables
ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app

EXPOSE 8000

# 8. Start Uvicorn bound to 0.0.0.0 so Fargate can see it
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]