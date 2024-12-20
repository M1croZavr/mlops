import os

from dotenv import load_dotenv
from minio import Minio

load_dotenv()  # Load environment variables from .env file if exists

ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")

minio_host = "minio"
minio_port = os.getenv("MINIO_PORT")
client = Minio(
    f"{minio_host}:{minio_port}",
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False,
)

sources_bucket_name = "sources"
datasets_bucket_name = "datasets"
artifacts_bucket_name = "artifacts"
# Make sources bucket
if not client.bucket_exists(sources_bucket_name):
    client.make_bucket(sources_bucket_name)
# Make datasets bucket
if not client.bucket_exists(datasets_bucket_name):
    client.make_bucket(datasets_bucket_name)
# Make artifacts bucket
if not client.bucket_exists(artifacts_bucket_name):
    client.make_bucket(artifacts_bucket_name)
