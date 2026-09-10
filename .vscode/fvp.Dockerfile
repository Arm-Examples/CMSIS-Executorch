# MDK FVP models for macOS hosts, which Arm does not build for.
#
# The image is the Linux build of the very same release the rest of the project
# pins in vcpkg-configuration.json (arm:models/arm/avh-fvp), so the model, the
# Ethos-U85 plugin and plugins/GDBServer.so are bit-identical to what a Linux or
# Windows developer gets from vcpkg. .vscode/fvp.sh builds it on first use.
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# 11.32.23 -> mdk-fvp-11.32_23_linux_arm64.tar.gz (fvp.sh derives and passes it)
ARG FVP_VERSION=11.32.23
ARG FVP_ARCHIVE=mdk-fvp-11.32_23_linux_arm64.tar.gz
ARG FVP_BASE_URL=https://artifacts.tools.arm.com/avh

# libpython is dlopen'd by the VSI/VIO bridges. We do not use them, but without
# a libpython3*.so the model prints a five-paragraph error before every run.
# The archive unpacks as ./<host>/{bin,plugins,...}, hence --strip-components 2.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl libatomic1 libpython3.10 && \
    ln -sf libpython3.10.so.1.0 "/usr/lib/$(uname -m)-linux-gnu/libpython3.10.so" && \
    curl -fsSL -o /tmp/fvp.tar.gz "${FVP_BASE_URL}/${FVP_VERSION}/${FVP_ARCHIVE}" && \
    mkdir -p /opt/avh-fvp && \
    tar -xf /tmp/fvp.tar.gz --strip-components 2 -C /opt/avh-fvp && \
    rm /tmp/fvp.tar.gz && \
    apt-get purge -y curl && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# The Arm user-based licence in ~/.armlm is bound to the account that activated
# it, so the container has to run as that same user name and uid or the model
# aborts with "The license found is assigned to another user".
ARG USERNAME=root
ARG USERID=0
RUN if [ "${USERID}" != "0" ]; then \
        groupadd -g "${USERID}" "${USERNAME}" && \
        useradd -l -r -u "${USERID}" -g "${USERNAME}" "${USERNAME}"; \
    fi
USER ${USERNAME}

ENV PATH="${PATH}:/opt/avh-fvp/bin"
ENV AVH_FVP_PLUGINS=/opt/avh-fvp/plugins
