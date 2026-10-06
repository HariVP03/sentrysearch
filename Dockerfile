FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install uv fastapi uvicorn httpx

COPY . .

RUN uv tool install --python 3.12 .

ENV PATH="/root/.local/bin:$PATH"
ENV HOME="/data"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
