# PhishEmail

# Setup

---

## Preparing the ingredients
1. Clone the repo
```commandline
git clone https://github.com/lindsey98/PhishEmail.git
```

2. Go to the project dir, download the models
First download from google drive
```commandline
file_id="YOUR_FILE_ID" && \
confirm=$(wget --quiet "https://drive.google.com/uc?export=download&id=${file_id}" -O- | sed -n 's/.*confirm=\(.*\)&amp;id.*/\1/p') && \
wget --load-cookies /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${confirm}&id=${file_id}" -O checkpoints.zip --continue
```

Then do
```commandline
zip_file="checkpoints.zip" && \
base_name=$(basename "$zip_file" .zip) && \
mkdir "$base_name" && \
unzip "$zip_file" -d "$base_name"
```

---

## Setup the Docker container
1. Build the docker image (This may take some time)
```commandline
sudo docker compose build
```
Make sure the docker image has been successfully built by verifying whether the image 'lindsey98/email' is listed in ``sudo docker images``.

2. Run the docker container
For Interactive mode (suppose we use a proxy)
```commandline
sudo docker compose up
```

For Detached mode
```commandline
sudo docker compose up -d
```

## Other useful commands
# View contents even if the container has been exited for some reasons
```commandline
sudo docker run --rm -it --entrypoint /bin/bash lindsey98/email
```

# Stop the container
```commandline
sudo docker stop lindsey98/email
```

# Prune unused docker images
```commandline
sudo docker system prune
```

