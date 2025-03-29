# Base image
FROM python:3.10-slim

# Install system dependencies (including libGL for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install LION from PyPI
RUN pip install --no-cache-dir --upgrade lionz

# Set working directory
WORKDIR /app

# Entry point for the MOOSE CLI
ENTRYPOINT ["lionz"]

# Default command
CMD ["-h"]
