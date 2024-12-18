FROM python:3.10-slim

COPY requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y curl
RUN python3 -m pip install -r requirements.txt

EXPOSE 5000
