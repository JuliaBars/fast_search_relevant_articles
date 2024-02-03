FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip3 install -r /app/requirements.txt --no-cache-dir

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]