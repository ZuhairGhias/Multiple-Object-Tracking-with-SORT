FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Native libraries required by opencv-python in headless/containerized environments.
RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.list.save \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
