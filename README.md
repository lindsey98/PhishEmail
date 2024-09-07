# PhishEmail

# Setup

---

## Preparing the ingredients
1. Clone the repo
```commandline

```

2. Download the models
```commandline
mkdir checkpoints
cd checkpoints/

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

