# Use the latest Ubuntu image
FROM ubuntu:latest

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and required tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    unzip \
    curl \
    poppler-utils \
    wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

# Add Deadsnakes PPA and install Python 3.8 with venv and dev packages
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as the default Python interpreter
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Create a Python 3.8 virtual environment and upgrade pip
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip

# Configure environment to use the virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the requirements file if available and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    pip install gdown && \
    rm /tmp/requirements.txt

# Configure generic git user for automation tasks
RUN git config --global user.name "automation-bot" && \
    git config --global user.email "automation-bot@example.com"

# Set working directory for repository cloning
WORKDIR /opt

# Clone the repository using generic placeholders
ARG GITHUB_TOKEN=""
ARG REPO_URL="https://github.com/USERNAME/REPOSITORY.git"
RUN if [ -z "$GITHUB_TOKEN" ]; then \
      git clone "$REPO_URL"; \
    else \
      git clone https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}; \
    fi

# Switch to the repository directory
ARG REPO_NAME="REPOSITORY"
WORKDIR /opt/${REPO_NAME}

# Use Bash for the following RUN command due to use of `shopt`
SHELL ["/bin/bash", "-c"]

# Download and set up checkpoints
RUN FILE_ID="1OqrhuKMOq8kjkKuU3jtPo9a-jA1zsJj8" && \
    FILE_NAME="checkpoints.zip" && \
    gdown --id "$FILE_ID" -O "$FILE_NAME" && \
    unzip "$FILE_NAME" -d checkpoints/ && \
    cd checkpoints && \
    if [ -d "checkpoints" ]; then \
      shopt -s dotglob && \
      mv checkpoints/* . && \
      shopt -u dotglob && \
      rmdir checkpoints; \
    fi && \
    cd ..

# Revert to default shell for future commands
SHELL ["/bin/sh", "-c"]

# Set default command to bash for interactive debugging or further commands
CMD ["bash"]
