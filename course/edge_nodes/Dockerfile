FROM flwr/superexec:1.29.0

USER root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
USER app
