# --- START OF MERGED DOCKERFILE ---

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# (Original copyright notice retained from provided Dockerfile)
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

# Base image selection:
# Choosing nvcr.io/nvidia/pytorch:21.06-py3 to meet the requirements:
# - Includes PyTorch
# - Includes CUDA 11.3
# - Uses Python 3 (specifically 3.8 in this image)
# - Does NOT include TensorFlow by default
# This image is also contemporary with DGL v0.7.0, improving build compatibility.
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3

# Stage 1: Build DGL from source
# Uses the selected base image
FROM ${FROM_IMAGE_NAME} AS dgl_builder

LABEL maintainer="NVIDIA"
LABEL description="Build stage for DGL v0.7.0 with CUDA 11.3 support"

# Install build dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential python3-dev make cmake \
    && rm -rf /var/lib/apt/lists/*

# Clone and checkout DGL v0.7.0
WORKDIR /dgl
RUN git clone --branch v0.7.0 --recurse-submodules --depth 1 https://github.com/dmlc/dgl.git .

# Modify CMakeLists to target appropriate CUDA compute capabilities
# Original: "35 50 60 70" (Kepler, Maxwell, Pascal, Volta)
# Modified: "60 70 80" (Pascal, Volta, Ampere) - Ampere (8.x) is supported by CUDA 11.3
RUN sed -i 's/"35 50 60 70"/"60 70 80"/g' cmake/modules/CUDA.cmake

# Configure and build DGL with CUDA and FP16 support
WORKDIR /dgl/build
RUN cmake -DUSE_CUDA=ON -DUSE_FP16=ON ..
RUN make -j$(nproc) # Use all available cores for faster build

# Stage 2: Final image
FROM ${FROM_IMAGE_NAME}

LABEL maintainer="Your Name <you@example.com>"
LABEL description="SE3 Transformer environment with custom-built DGL (v0.7.0) for CUDA 11.3"

# Set the working directory INSIDE the container
WORKDIR /workspace/app # Changed name slightly for clarity

# Copy requirements first (from the root context)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt

# --- THIS IS THE KEY CHANGE ---
# Copy ONLY the contents of NEW-SE3Transformer from the (pruned) build context
# into the current WORKDIR (/workspace/app) inside the image.
COPY NEW-SE3Transformer/ .

# Set environment variables required by the application/DGL
ENV DGLBACKEND=pytorch
ENV OMP_NUM_THREADS=1

# Optional: Define entrypoint or default command if needed
# CMD ["python", "your_script.py"] # Adjust script path if needed

# --- END OF MERGED DOCKERFILE ---