#!/usr/bin/env bash
# Stand-in for the FVP_Corstone_SSE-320 executable, referenced from the
# `debugger: model:` node of the SSE-320-U85 target-set. The CMSIS Solution
# extension runs it both as the "CMSIS Run" task and as the gdbtarget debug
# server, in the latter case with
#
#   -D --plugin $AVH_FVP_PLUGINS/GDBServer.so -C GDBServer.port=3333 \
#      -f board/Corstone-320/fvp_config.txt --simlimit 60 -a <application>.hex
#
# It does two things the bare model command cannot:
#
#   * resolves plugins/GDBServer.so itself. The generated launch config takes
#     the path from $AVH_FVP_PLUGINS, which only exists in a shell that has run
#     `vcpkg activate` -- VS Code, started from Finder or Spotlight, has not.
#   * runs the model in Docker on macOS, where Arm publishes no FVP build, with
#     the GDB port forwarded to the host so arm-none-eabi-gdb can reach it.
#
# On Linux it just execs the real model. On Windows, where the extension cannot
# run a bash script, point the csolution's `model:` at FVP_Corstone_SSE-320.exe.
set -euo pipefail

MODEL="${FVP_MODEL:-FVP_Corstone_SSE-320}"
FVP_VERSION="${FVP_VERSION:-11.32.23}"          # keep in sync with vcpkg-configuration.json
IMAGE="${FVP_IMAGE:-cmsis-fvp:${FVP_VERSION}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_macos() { [[ "$(uname -s)" == "Darwin" ]]; }

# Where GDBServer.so lives, given that $AVH_FVP_PLUGINS is often unset.
plugin_dir() {
    if is_macos; then
        echo "/opt/avh-fvp/plugins"                       # inside the container
        return
    fi
    if [[ -n "${AVH_FVP_PLUGINS:-}" && -d "${AVH_FVP_PLUGINS}" ]]; then
        echo "${AVH_FVP_PLUGINS%/}"
        return
    fi
    local model_path
    model_path="$(command -v "${MODEL}" 2>/dev/null || true)"
    [[ -n "${model_path}" ]] && echo "$(dirname "$(dirname "$(realpath "${model_path}")")")/plugins"
}

# Rewrite `--plugin <path>` when <path> does not resolve, and pick the GDB port
# out of `-C GDBServer.port=<n>` so the macOS branch knows what to forward.
ARGS=()
GDB_PORT=""
ALLOW_REMOTE_SET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plugin)
            plugin="${2:-}"
            if [[ ! -f "${plugin}" ]] || is_macos; then
                dir="$(plugin_dir)"
                [[ -n "${dir}" ]] && plugin="${dir}/$(basename "${plugin}")"
            fi
            ARGS+=("--plugin" "${plugin}")
            shift 2
            ;;
        -C)
            [[ "${2:-}" == GDBServer.port=* ]] && GDB_PORT="${2#GDBServer.port=}"
            [[ "${2:-}" == GDBServer.allow_remote=* ]] && ALLOW_REMOTE_SET=1
            ARGS+=("-C" "${2:-}")
            shift 2
            ;;
        -C*)
            [[ "$1" == -CGDBServer.port=* ]] && GDB_PORT="${1#-CGDBServer.port=}"
            [[ "$1" == -CGDBServer.allow_remote=* ]] && ALLOW_REMOTE_SET=1
            ARGS+=("$1")
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# The model block-buffers its own stdout whenever it is a pipe -- which it
# always is under VS Code -- so "GDBServer: Listening ... port=3333" and every
# semihosting printf sit in a 4KB buffer until the model exits. The debug
# adapter waits for that banner to know the server is up (target.serverPortRegExp
# in launch.json), so without line buffering it gives up and kills the model
# just before the line it was waiting for gets flushed.
RUNNER=()
command -v stdbuf >/dev/null 2>&1 && RUNNER=(stdbuf -oL -eL)

if ! is_macos; then
    command -v "${MODEL}" >/dev/null 2>&1 || {
        echo "${MODEL} not found on PATH. Run 'vcpkg activate' in this folder," >&2
        echo "or set FVP_MODEL to the full path of the model executable." >&2
        exit 1
    }
    exec "${RUNNER[@]}" "${MODEL}" "${ARGS[@]}"
fi

# ---------------------------------------------------------------- macOS/Docker
docker info >/dev/null 2>&1 || {
    echo "Docker is not running. On macOS the FVP runs in a container because" >&2
    echo "Arm ships no macOS build of the model." >&2
    exit 1
}

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    case "$(uname -m)" in
        arm64|aarch64) arch="arm64" ;;
        *)             arch="x86" ;;
    esac
    archive="mdk-fvp-${FVP_VERSION%.*}_${FVP_VERSION##*.}_linux_${arch}.tar.gz"
    echo "Building ${IMAGE} from ${archive} (first run only, ~100MB download)."
    echo "By using these models you accept the Arm FVP End User License Agreement:"
    echo "  https://artifacts.tools.arm.com/avh/${FVP_VERSION}/license_agreement.txt"
    docker build -t "${IMAGE}" -f "${HERE}/fvp.Dockerfile" \
        --build-arg "FVP_VERSION=${FVP_VERSION}" \
        --build-arg "FVP_ARCHIVE=${archive}" \
        --build-arg "USERNAME=$(whoami)" \
        --build-arg "USERID=$(id -u)" \
        "${HERE}" >&2
fi

PORTS=()
NAME=()
if [[ -n "${GDB_PORT}" ]]; then
    PORTS=(-p "${GDB_PORT}:${GDB_PORT}")
    # GDBServer binds to localhost unless told otherwise, and a connection
    # arriving through Docker's port forwarder is not localhost as far as the
    # container is concerned -- without this GDB's first packet is met with a
    # connection reset.
    [[ "${ALLOW_REMOTE_SET}" == "0" ]] && ARGS+=("-C" "GDBServer.allow_remote=1")
    # A container left behind by a debug session that ended badly still holds
    # the port; the next launch would fail on bind rather than on connect.
    NAME=(--name "cmsis-fvp-${GDB_PORT}")
    docker rm -f "cmsis-fvp-${GDB_PORT}" >/dev/null 2>&1 || true
fi

# $HOME is bind-mounted at the same path so the ELF, the config file and
# ~/.armlm are reachable under the paths the extension passes in.
MOUNTS=(--mount "type=bind,src=${HOME}/,dst=${HOME}/")
[[ "${PWD}" == "${HOME}"* ]] || MOUNTS+=(--mount "type=bind,src=${PWD}/,dst=${PWD}/")

exec docker run --rm -i --init "${NAME[@]}" "${PORTS[@]}" "${MOUNTS[@]}" \
    --workdir "${PWD}" \
    --env "ARMLM_CACHED_LICENSES_LOCATION=${HOME}/.armlm" \
    "${IMAGE}" stdbuf -oL -eL "${MODEL}" "${ARGS[@]}"
