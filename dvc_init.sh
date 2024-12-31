source .env

poetry run dvc init

echo "dvc initialized"

poetry run dvc remote add -d minio s3://sources
poetry run dvc remote modify minio access_key_id "$MINIO_ROOT_USER"
poetry run dvc remote modify minio secret_access_key "$MINIO_ROOT_PASSWORD" 
poetry run dvc remote modify minio endpointurl http://localhost:9000
poetry run dvc remote modify minio use_ssl false

echo "s3://sources set up as remote"