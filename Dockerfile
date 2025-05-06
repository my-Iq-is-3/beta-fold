# --- START OF SIMPLIFIED DOCKERFILE (Using Pip Install) ---

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# (Original copyright notice retained)
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

# Base image selection: nvcr.io/nvidia/pytorch:21.06-py3
# - PyTorch, CUDA 11.3, Python 3.8
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3
FROM ${FROM_IMAGE_NAME}

LABEL maintainer="Zach C <your_email@example.com>"
LABEL description="SE3 Transformer environment with DGL v0.7.0 (cu113) via pip"

ENV DEBIAN_FRONTEND=noninteractive

# --- FIX: Install bash and set as default shell ---
RUN apt-get update && apt-get install -y --no-install-recommends bash && \
    rm -rf /var/lib/apt/lists/* # Clean up apt cache
RUN chsh -s /bin/bash root

# Set the working directory INSIDE the container
WORKDIR /workspace/app

# Copy requirements first (ensure DGL v0.7.0 for cu113 is listed inside)
COPY requirements.txt .
# This will now install DGL along with other requirements
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt

# Copy the application code
COPY NEW-SE3Transformer/ .

# Set environment variables required by the application/DGL
ENV DGLBACKEND=pytorch
ENV OMP_NUM_THREADS=1

# No CMD or ENTRYPOINT needed, will use Paperspace's "Command" field

# --- END OF SIMPLIFIED DOCKERFILE ---