#!/bin/bash

cd datasets/
rm phishing-links-ACTIVE-TODAY.txt
# Setting up retry variables
max_retries=5
retry_delay=10  # Time delay in seconds between retries

# Retry loop for wget
for (( i=1; i<=max_retries; i++ )); do
    https_proxy=127.0.0.1:7890 wget https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE-TODAY.txt
    if [ $? -eq 0 ]; then
        echo "Download successful."
        break
    else
        echo "Download failed. Retrying in $retry_delay seconds... (Attempt $i of $max_retries)"
        sleep $retry_delay
    fi
done
cd ../