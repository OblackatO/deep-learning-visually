# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Pedro Gomes, 2026: renamed the working directory from /dli to
# /course and added a pointer to Dockerfile.cpu.

# PyTorch + CUDA 11.8 + cuDNN 8 — requires NVIDIA GPU + nvidia-container-toolkit on the host
# Run with: docker run --gpus all ...
# On macOS or machines without an NVIDIA GPU, use Dockerfile.cpu instead.
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System deps (Pillow needs libjpeg/libpng)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-turbo8 \
        libpng16-16 \
        git \
        gcc \
    && rm -rf /var/lib/apt/lists/*
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev

COPY ./environment/requirements.txt .
RUN pip install -r requirements.txt


RUN pip install --upgrade jupyterlab==4.6.2
RUN pip install ipywidgets==8.1.2

ENV SHELL=/bin/bash
ENV PYTHONPATH=/course
WORKDIR /course/
COPY . .

WORKDIR /course/tutorials

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--allow-root", \
     "--no-browser", \
     "--NotebookApp.token=", \
     "--NotebookApp.password="]
