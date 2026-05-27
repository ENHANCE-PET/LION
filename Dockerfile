# Base image
FROM python:3.10-slim

ARG LIONZ_VERSION

# Install system dependencies (including libGL for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install the exact LIONZ release from PyPI.
RUN python -m pip install --no-cache-dir "lionz==${LIONZ_VERSION}"

# Set working directory
WORKDIR /app

# Entry point for the LIONZ CLI
ENTRYPOINT ["lionz"]

# Default command
CMD ["-h"]
