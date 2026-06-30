FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
# LLM-ключ передаётся через env при запуске:
#   docker run -e LLM_API_KEY=... -e LLM_MODEL=gpt-4o-mini -p 8000:8000 pora-ai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
