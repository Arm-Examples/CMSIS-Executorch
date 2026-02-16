FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA for Python 3.13 (not in Ubuntu 24.04 default repos)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    rm -rf /var/lib/apt/lists/*

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
    python3.13 \
    python3.13-dev \
    python3.13-venv \
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

# Install current CMake from Kitware's official repository
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    echo 'deb https://apt.kitware.com/ubuntu/ noble main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.13 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

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
    git checkout v1.1.0 && \
    git submodule sync && \
    git submodule update --init --recursive

# Create CMake module to patch CMSIS-NN for CMake 4.x compatibility
RUN mkdir -p /workspace/executorch/cmake/modules && \
    cat > /workspace/executorch/cmake/modules/PatchCmsisNN.cmake << 'EOF'
# Patch CMSIS-NN after FetchContent populates it for CMake 4.x compatibility
macro(patch_cmsis_nn_if_needed)
  if(EXISTS "${cmsis_nn_SOURCE_DIR}/CMakeLists.txt")
    file(READ "${cmsis_nn_SOURCE_DIR}/CMakeLists.txt" CMSIS_NN_CONTENT)
    string(FIND "${CMSIS_NN_CONTENT}" "target_include_directories(cmsis-nn PUBLIC \"Include\")" HAS_RELATIVE_PATH)
    if(NOT HAS_RELATIVE_PATH EQUAL -1)
      message(STATUS "Patching CMSIS-NN for CMake 4.x compatibility")
      string(REPLACE 
        "target_include_directories(cmsis-nn PUBLIC \"Include\")"
        "target_include_directories(cmsis-nn PUBLIC \"\${CMAKE_CURRENT_SOURCE_DIR}/Include\")"
        CMSIS_NN_CONTENT "${CMSIS_NN_CONTENT}")
      file(WRITE "${cmsis_nn_SOURCE_DIR}/CMakeLists.txt" "${CMSIS_NN_CONTENT}")
    endif()
  endif()
endmacro()
EOF

# Patch ExecuTorch's cortex_m backend to use the patching module
RUN cd /workspace/executorch/backends/cortex_m && \
    sed -i '/FetchContent_MakeAvailable(cmsis_nn)/i\  include(${CMAKE_SOURCE_DIR}/cmake/modules/PatchCmsisNN.cmake)\n  FetchContent_Populate(cmsis_nn)\n  patch_cmsis_nn_if_needed()' CMakeLists.txt && \
    sed -i 's/FetchContent_MakeAvailable(cmsis_nn)/add_subdirectory(${cmsis_nn_SOURCE_DIR} ${cmsis_nn_BINARY_DIR})/' CMakeLists.txt

RUN python -m venv executorch-venv
ENV PATH="/workspace/executorch/examples/arm/arm-scratch/arm-gnu-toolchain-13.3.rel1-arm-none-eabi/bin:/workspace/executorch-venv/bin:$PATH"

# Use ExecuTorch's official installation script but avoid ATen bridge conflicts
RUN pip install --extra-index-url https://download.pytorch.org/whl/test/cpu --break-system-packages \
                executorch==1.1.0 \
                ethos-u-vela==4.5.0 \
                torch==2.10.0+cpu \
                torchvision==0.25.0+cpu \
                torchao==0.15.0+cpu \
                torchaudio==2.10.0+cpu

# Remove visualization-only model-explorer packages from requirements files
# (not needed for compilation, and they all require Python <3.13)
RUN sed -i '/model-explorer/d' \
    /workspace/executorch/backends/arm/requirements-arm-tosa.txt \
    /workspace/executorch/backends/arm/requirements-arm-ethos-u.txt

# Setup ARM environment using official script
WORKDIR /workspace/executorch
RUN ./examples/arm/setup.sh --i-agree-to-the-contained-eula

# Rename architecture-specific toolchain folder to neutral name for multi-arch host-support
RUN cd examples/arm/arm-scratch && \
    TOOLCHAIN_DIR=$(ls -d arm-gnu-toolchain-13.3.rel1-*-arm-none-eabi 2>/dev/null | head -1) && \
    if [ -n "$TOOLCHAIN_DIR" ] && [ "$TOOLCHAIN_DIR" != "arm-gnu-toolchain-13.3.rel1-arm-none-eabi" ]; then \
        echo "Renaming $TOOLCHAIN_DIR to arm-gnu-toolchain-13.3.rel1-arm-none-eabi" && \
        mv "$TOOLCHAIN_DIR" arm-gnu-toolchain-13.3.rel1-arm-none-eabi; \
    else \
        echo "Toolchain already has neutral name or not found"; \
    fi

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