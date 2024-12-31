if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <file_or_dir_path> <dataset_name> <version>"
  exit 1
fi

FILE_PATH="$1"
DATASET_NAME="$2"
VERSION="$3"

poetry run dvc add "$FILE_PATH"

echo "Successfully added '$FILE_PATH' to DVC."
read -p "Do you want to proceed with committing and pushing changes? (y/n): " CONFIRMATION

if [[ "$CONFIRMATION" == "n" || "$CONFIRMATION" == "N" ]]; then
  echo "Reverting the added file from DVC..."
  poetry run dvc remove "$FILE_PATH.dvc"
  echo "File '$FILE_PATH' removed from DVC."
  exit 0
fi

poetry run dvc commit

git add "$FILE_PATH.dvc"
git commit -m "data vers: $DATASET_NAME v.$VERSION"

poetry run dvc push

echo "Changes committed and data pushed successfully!"