# --- START OF FIXED DOCKERFILE (Scenario B) ---

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# (Original copyright notice retained)
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

# Base image selection: nvcr.io/nvidia/pytorch:21.06-py3
# - PyTorch, CUDA 11.3, Python 3.8
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3

# -------------------- Stage 1: Build DGL from source --------------------
FROM ${FROM_IMAGE_NAME} AS dgl_builder

LABEL maintainer="NVIDIA"
LABEL description="Build stage for DGL v0.7.0 with CUDA 11.3 support"

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies AND bash (useful for debugging this stage if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    build-essential \
    python3-dev \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set bash as the default shell for root (useful for debugging this stage if needed)
RUN chsh -s /bin/bash root

# Clone and checkout DGL v0.7.0
WORKDIR /dgl_source # Changed WORKDIR to avoid potential conflicts
RUN git clone --branch v0.7.0 --recurse-submodules --depth 1 https://github.com/dmlc/dgl.git .

# Modify CMakeLists to target appropriate CUDA compute capabilities (Pascal, Volta, Ampere)
RUN sed -i 's/"35 50 60 70"/"60 70 80"/g' cmake/modules/CUDA.cmake

# Configure and build DGL with CUDA and FP16 support
WORKDIR /dgl_source/build
RUN cmake -DUSE_CUDA=ON -DUSE_FP16=ON ..
RUN make -j$(nproc) # Use all available cores for faster build

# --- Install the built DGL into this stage's /usr/local ---
RUN make install

# -------------------- Stage 2: Final image --------------------
FROM ${FROM_IMAGE_NAME}

LABEL maintainer="Zach C <your_email@example.com>" # Updated maintainer
LABEL description="SE3 Transformer environment with custom-built DGL (v0.7.0) for CUDA 11.3"

ENV DEBIAN_FRONTEND=noninteractive

# --- FIX: Install bash and set as default shell in the FINAL stage ---
RUN apt-get update && apt-get install -y --no-install-recommends bash && \
    rm -rf /var/lib/apt/lists/* # Clean up apt cache
RUN chsh -s /bin/bash root

# --- ADD: Copy the installed custom DGL from the builder stage ---
# This copies libraries, Python packages, headers etc. installed by `make install`
COPY --from=dgl_builder /usr/local/ /usr/local/

# Set the working directory INSIDE the container
WORKDIR /workspace/app

# Copy requirements first (ensure DGL is NOT listed here)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt

# Copy the application code
COPY NEW-SE3Transformer/ .

# Set environment variables required by the application/DGL
ENV DGLBACKEND=pytorch
ENV OMP_NUM_THREADS=1
# Python path might be needed depending on exact install location from make install vs base image's site-packages
# Consider uncommenting if Python can't find DGL:
# ENV PYTHONPATH=/usr/local/lib/python3.8/dist-packages:${PYTHONPATH} # Adjust python3.8/dist-packages if needed

# No CMD or ENTRYPOINT needed, will use Paperspace's "Command" field

# --- END OF FIXED DOCKERFILE ---
