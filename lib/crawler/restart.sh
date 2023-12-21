#!/bin/bash

# Name for the Docker container
CONTAINER_NAME="splash-container"

# Infinite loop
while true; do
    # Start the Splash container
    echo "Starting Splash container..."
    sudo docker run --name $CONTAINER_NAME -d -p 8050:8050 scrapinghub/splash

    # Wait for a specified time (e.g., 10 minutes)
    sleep 600

    # Stop and remove the container
    echo "Stopping and removing Splash container..."
    sudo docker stop $CONTAINER_NAME
    sudo docker rm $CONTAINER_NAME

    # Optional: Short delay before restarting the loop
    sleep 5
done
