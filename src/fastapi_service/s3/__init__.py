import os

from dotenv import load_dotenv
from minio import Minio

load_dotenv()  # Load environment variables from .env file

ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")

client = Minio(
    "127.0.0.1:9000",
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False,
)

datasets_bucket_name = "datasets"
artifacts_bucket_name = "artifacts"
# Make datasets bucket
if not client.bucket_exists(datasets_bucket_name):
    client.make_bucket(datasets_bucket_name)
# Make artifacts bucket
if not client.bucket_exists(artifacts_bucket_name):
    client.make_bucket(artifacts_bucket_name)
