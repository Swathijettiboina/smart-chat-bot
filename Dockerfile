FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# COPY . /app/
RUN git clone https://github.com/Swathijettiboina/Smart-PDF-Chat-Assistant.git .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "bot.py", "--server.port=8501", "--server.address=0.0.0.0"]