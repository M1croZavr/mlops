import os

from dotenv import load_dotenv
from minio import Minio

load_dotenv()  # Load environment variables from .env file if exists

ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")

minio_host = "minio"
minio_port = os.getenv("MINIO_PORT", 9000)
client = Minio(
    f"{minio_host}:{minio_port}",
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False,
)
SOURCES_BUCKET_NAME = "sources"
DATASETS_BUCKET_NAME = "datasets"
ARTIFACTS_BUCKET_NAME = "artifacts"
