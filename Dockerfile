FROM python:3.12.2-slim

WORKDIR /app

COPY 대구공고전문.txt api_stream.py unstruct_step1.py requirements.txt /app/
COPY vector_db /app/vector_db
COPY templates /app/templates
COPY static /app/static

RUN apt-get update && \
    apt-get install -y openjdk-17-jdk build-essential gcc libpq-dev && \
    apt-get clean

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "api_stream:app", "--host", "0.0.0.0", "--port", "6677"]