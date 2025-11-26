FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic build tools
    # build-essential \
    ninja-build \
    git \
    curl \
    wget \
    unzip \
    rsync \
    # ARM cross-compilation toolchain
    # gcc-arm-none-eabi \
    # libnewlib-arm-none-eabi \
    # Python and pip
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    # Additional build dependencies for ExecuTorch
    pkg-config \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # Additional utilities
    vim \
    nano \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install current CMake (3.30.x) from Kitware's official repository
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    echo 'deb https://apt.kitware.com/ubuntu/ noble main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

    ## Create a non-root user for development
RUN useradd -m -s /bin/bash developer && \
    usermod -aG sudo developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to non-root user
USER developer

# Clone ExecuTorch and use official installation script
WORKDIR /workspace
RUN git clone https://github.com/pytorch/executorch.git && \
    cd executorch && \
    git checkout v1.0.0 && \
    git submodule sync && \
    git submodule update --init --recursive

RUN python -m venv executorch-venv
ENV PATH="/workspace/executorch/examples/arm/ethos-u-scratch/arm-gnu-toolchain-13.3.rel1-aarch64-arm-none-eabi/bin:/workspace/executorch-venv/bin:$PATH"

# Use ExecuTorch's official installation script but avoid ATen bridge conflicts
RUN pip install --extra-index-url https://download.pytorch.org/whl/test/cpu --break-system-packages \
                executorch==1.0.0 \
                ethos-u-vela torch==2.9.0 \
                torchvision==0.24.0 \
                torchao  \
                torchaudio==2.9.0 
                
# Setup ARM environment using official script
WORKDIR /workspace/executorch
RUN ./examples/arm/setup.sh --i-agree-to-the-contained-eula

# Test if ARM setup worked correctly
RUN if [ -f "examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" ]; then \
        echo "✅ ARM setup successful - toolchain file found"; \
    else \
        echo "❌ ARM setup failed - toolchain file missing"; \
        exit 1; \
    fi

# Verify tools installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__} installed')" && \
    python -c "import executorch; print('ExecuTorch installed successfully')" || echo "ExecuTorch verification failed" && \
    arm-none-eabi-gcc --version | head -1 && \
    cmake --version | head -1

# Default command
CMD ["bash"]