# 1️⃣ Use lightweight Python image
FROM python:3.9-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements first (better caching)
COPY requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy application code
COPY api ./api
COPY models ./models

# 6️⃣ Expose FastAPI port
EXPOSE 8000

# 7️⃣ Run FastAPI with Uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
