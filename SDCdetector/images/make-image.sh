#!/usr/bin/env bash
# Usage: bash update_enroot_image.sh INPUT.sqsh [OUTPUT.sqsh]
set -euo pipefail

INPUT="base-pytorch-26.09-py3.sqsh"
OUTPUT="pytorch-env-v2.sqsh"
NOTE="Pytorch base image 26.09 of NVIDIA NGC plus installation of NCCL Inspector Plugin (Version of NCCL Repo commit '3e46c089f4ae4732722a419eb6096aead7e47a30 - Thu Oct 8 14:16:00 2026 -0700 - Justus Calvin') to help observe NVLS behavior. Plus installation of FlashAttention 2, 3 and 4 for CUDA 13.x and 12.x. Also includes Prometheus client to parse NCCL Inspector metrics."

HOST_DIR=$(realpath -e -- "${HOST_DIR:-./inspector}")
CONTAINER_DIR="/opt/nccl-inspector"

[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || {
    printf 'Error: Output already exists.\n' >&2
    exit 1
}

STAGING=$(mktemp -d "./.enroot-export.XXXXXX")
trap 'rm -rf -- "$STAGING"' EXIT
printf 'rc() { exec "$@"; }\n' > "$STAGING/build.conf"

printf 'Unpack: %s\n' "$INPUT"
enroot create --name build "$INPUT"

printf 'Modifying the container ...\n'
enroot start --root --rw --conf "$STAGING/build.conf" \
    --mount "$HOST_DIR:/mnt/image-import:none:bind,ro,x-create=dir" \
    --env "CONTAINER_DIR=$CONTAINER_DIR" --env "IMAGE_NOTE=$NOTE" \
    build /bin/bash -s <<'CONTAINER'
set -euo pipefail
cd /
# ===== Put modifications here =====
# Prometheus client to parse NCCL Inspector metrics
python -m pip install prometheus-client==0.26.0

# Build-tools and dependencies for FlashAttention
python -m pip install packaging psutil ninja einops setuptools wheel

# FlashAttention 3 from official repo (FA2 and FA4 are already included in the base image, FA1 is not important here since its not compatible with CUDA 13.x)
python -m pip install --no-build-isolation "git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=hopper"

# Copy the host folder into the image
mkdir -p -- "$CONTAINER_DIR"
cp -R --no-preserve=all -- /mnt/image-import/. "$CONTAINER_DIR/"
printf '%s\n' "$IMAGE_NOTE" > /etc/image-note.txt

# Building the NCCL Inspector Plugin
cd $CONTAINER_DIR
make

mkdir -p /dumps # Create a directory for NCCL Inspector dumps

# print path to libnccl-profiler-inspector.so
echo "Path to libnccl-profiler-inspector.so: $CONTAINER_DIR/libnccl-profiler-inspector.so"

# export NCCL_PROFILER environment variables
echo "export NCCL_PROFILER_PLUGIN=$CONTAINER_DIR/libnccl-profiler-inspector.so" >> /etc/profile.d/nccl-inspector.sh
echo "export NCCL_INSPECTOR_DUMP_DIR=/dumps" >> /etc/profile.d/nccl-inspector.sh

# ===== End of modifications =====
CONTAINER

printf 'Exporting new Image ...\n'
enroot export --output "$STAGING/image.sqsh" build
[[ -s "$STAGING/image.sqsh" ]] || {
    printf 'Error: Export failed, no image created.\n' >&2
    exit 1
}
ln -T -- "$STAGING/image.sqsh" "$OUTPUT"
enroot remove --force build
printf 'Done: %s\n' "$OUTPUT"