RETRY_COUNT=3
ENV_NAME="${ENV_NAME:-emailenv}" # by default is emailenv

download_with_retry() {
  local file_id="$1"
  local file_name="$2"
  local count=0
  until [ $count -ge $RETRY_COUNT ]
  do
    echo "Attempting to download $file_name (Attempt $((count + 1))/$RETRY_COUNT)..."
    conda run -n "$ENV_NAME" gdown --id "$file_id" -O "$file_name" && break
    count=$((count + 1))
    echo "Retry $count of $RETRY_COUNT for $file_name..."
    sleep 2  # Increased wait time to 2 seconds
  done

  if [ $count -ge $RETRY_COUNT ]; then
    error_exit "Failed to download $file_name after $RETRY_COUNT attempts."
  fi
}

# System dependencies: For Linux (required by pdf2image)
sudo apt-get install poppler-utils
sudo apt-get install wkhtmltopdf

# Initialize Conda for bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
# Check if the environment exists
if conda info --envs | grep -w "^$ENV_NAME" > /dev/null 2>&1; then
  echo "Activating existing Conda environment: $ENV_NAME"
else
  echo "Creating new Conda environment: $ENV_NAME with Python 3.8"
  conda create -y -n "$ENV_NAME" python=3.8
fi

# Activate the Conda environment
echo "Activating Conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Install Python dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
  echo "Installing Python dependencies from requirements.txt..."
  conda run -n "$ENV_NAME" pip install -r requirements.txt
else
  error_exit "requirements.txt not found in the current directory."
fi

# Download checkpoints
FILE_ID = "1OqrhuKMOq8kjkKuU3jtPo9a-jA1zsJj8"
FILE_NAME = "checkpoints.zip"
download_with_retry "$FILE_ID" "$FILE_NAME"

unzip "$FILE_NAME" -d checkpoints/
cd checkpoints || exit 1  # Exit if the directory doesn't exist

# Check if there's a nested 'checkpoints/' directory
if [ -d "checkpoints" ]; then
  echo "Nested directory 'checkpoints/' detected. Moving contents up..."
  shopt -s dotglob
  # Move everything from the nested directory to the current directory
  mv checkpoints/* .
  shopt -u dotglob
  rmdir checkpoints
  cd ../
else
  echo "No nested 'checkpoints/' directory found. No action needed."
fi

echo "All packages installed and models downloaded successfully!"
