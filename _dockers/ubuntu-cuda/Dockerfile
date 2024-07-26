# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.1.1


FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.10

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND="noninteractive" \
    TZ="Etc/UTC" \
    PATH="$PATH:/root/.local/bin" \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MKL_THREADING_LAYER="GNU" \
    # MAKEFLAGS="-j$(nproc)"
    MAKEFLAGS="-j4"

RUN \
    apt-get -y update --fix-missing && \
    apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        build-essential \
        ca-certificates \
        software-properties-common \
        nvidia-cuda-toolkit \
        pkg-config \
        libopenmpi-dev \
        openmpi-bin \
        cmake \
        git \
        git-lfs \
        wget \
        curl \
        unzip \
        g++ \
        cmake \
        ffmpeg \
        git \
        ssh \
        tree \
    && \
    # Install python
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-dev \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="/usr/lib/python${PYTHON_VERSION}/site-packages"

COPY ./ repo/

RUN \
    CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
    CU_VERSION_MM=${CUDA_VERSION_MM//'.'/''} && \
    cd repo && \
    pip install --no-cache-dir -r requirements.txt && \
    python .actions/assistant.py aggregate-requirements "./*/requirements.txt" "./_requirements/base.txt" && \
    cat _requirements/base.txt && \
    pip install pycuda numba --no-cache-dir -r _requirements/base.txt \
      --extra-index-url="https://download.pytorch.org/whl/cu${CU_VERSION_MM}" && \
    cd .. && \
    rm -rf repo

RUN \
    # Show what we have
    pip --version && \
    pip list && \
    nvcc --version && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; assert torch.__version__.startswith('2.'), torch.__version__"
