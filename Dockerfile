FROM python:3.10-slim
WORKDIR /work
COPY . /work
RUN pip install --no-cache-dir -r requirements.txt
# Default arguments can be specified here, but can be overridden
EXPOSE 8000
ENTRYPOINT ["python", "-m", "inference"]
CMD ["--email_dir=./test_emails/2015"]