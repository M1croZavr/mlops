FROM python:3.10-bullseye

RUN pip3 install poetry==1.8.2
RUN pip3 install cmake

WORKDIR /app

COPY pyproject.toml poetry.lock .env README.md ./
COPY src ./src
COPY tests ./tests

RUN poetry install --without dev

ENTRYPOINT ["poetry", "run", "start-server"]
