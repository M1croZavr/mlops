FROM python:3.10

WORKDIR /app

COPY README.md ./
COPY tests ./tests

COPY pyproject.toml poetry.lock ./
RUN pip3 install poetry==1.8.2
RUN poetry install --without dev --no-root

COPY src ./src
RUN poetry install --only-root

COPY data/MNIST.tar.gz ./data/MNIST.tar.gz

EXPOSE 8000

ENTRYPOINT ["poetry", "run", "start-server"]
