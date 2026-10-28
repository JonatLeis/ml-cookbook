#!/usr/bin/env bash
#SBATCH --job-name=sdc-suite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=slurmlogs/exp-%j.log
#SBATCH --error=slurmlogs/exp-%j.err

set -euo pipefail
umask 077

# Environment variables, get default values if not set
export LOGGING="${LOGGING:-}"                               # Should be set in config.yml but can be overridden here
export API_PORT="${API_PORT:-}"                             # Should be set in config.yml but can be overridden here
export API_HOST="${API_HOST:-}"                             # Should be set in config.yml but can be overridden here
export API_LINGER="${API_LINGER:-}"                         # Should be set in config.yml but can be overridden here
export PROFILER_ACTIVE="${PROFILER_ACTIVE:-0}"              # Enable nsys profiler, only recommended during debug since it introduces significant overhead
export NCCL_INSPECTOR_ENABLE="${NCCL_INSPECTOR_ENABLE:-0}"  # Enable NCCL Inspector, its recommended to enable nccl inspector and nvls together and set the appropriate NCCL_ALGO to assert nvls is being used.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"            # Enable NVLink sharp, its recommended to enable nccl inspector and nvls together and set the appropriate NCCL_ALGO to assert nvls is being used.
export NCCL_ALGO="${NCCL_ALGO:-allreduce:NVLS}"             # Set NCCL Algorithms, its recommended to enable nccl inspector and nvls together and set the appropriate NCCL_ALGO to assert nvls is being used.
export NCCL_INSPECTOR_DUMP_DIR="${NCCL_INSPECTOR_DUMP_DIR:-/dumps}"
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS="${NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS:-30000000}"
export NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES="${NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES:-0}"
export NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING="${NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING:-0}"
export NCCL_INSPECTOR_OTEL_EXPORT="${NCCL_INSPECTOR_OTEL_EXPORT:-0}"

# container mounts and directories belong here. Application specific settings in config.yml!
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?}"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/config.yml}"
IMAGE="$PROJECT_ROOT/images/pytorch-env-v2.sqsh"
RESULTS="$PROJECT_ROOT/runs/expID_$SLURM_JOB_ID"
PROFILE_HOST="$PROJECT_ROOT/profiles/run-$SLURM_JOB_ID"
DUMP_DIR="$PROJECT_ROOT/dumps/run-$SLURM_JOB_ID"
[[ -f "$CONFIG_FILE" ]] || { echo "Missing config: $CONFIG_FILE" >&2; exit 2; }
mkdir -p "$RESULTS" "$PROFILE_HOST" "$DUMP_DIR"

# Optional explicit environment overrides
export PROFILER_ACTIVE="${PROFILER_ACTIVE:-0}"
export NCCL_INSPECTOR_ENABLE="${NCCL_INSPECTOR_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_ALGO="${NCCL_ALGO:-allreduce:NVLS}"
export NCCL_PROFILER_PLUGIN=/opt/nccl-inspector/libnccl-profiler-inspector.so # hardcoded to match container build, not configurable
export NCCL_INSPECTOR_DUMP_DIR=/dumps
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=30000000
export NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES=0
export NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=0
export NCCL_INSPECTOR_OTEL_EXPORT=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1

srun \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 \
    --cpu-bind=cores --kill-on-bad-exit=1 --label \
    --container-image="$IMAGE" --container-name=sdc-container \
    --no-container-mount-home --no-container-entrypoint \
    --container-workdir=/workspace \
    --container-mounts="$RESULTS:/results,$PROJECT_ROOT/src:/workspace:ro,$CONFIG_FILE:/config.yml:ro,$PROFILE_HOST:/profiles,$DUMP_DIR:/dumps" \
    --container-env=LOGGING,API_PORT,API_HOST,API_LINGER,PYTHONNOUSERSITE,PYTHONDONTWRITEBYTECODE,PYTHONUNBUFFERED,CUBLAS_WORKSPACE_CONFIG,OMP_NUM_THREADS,PROFILER_ACTIVE,PROFILE,NCCL_NVLS_ENABLE,NCCL_ALGO,NCCL_PROFILER_PLUGIN,NCCL_INSPECTOR_ENABLE,NCCL_INSPECTOR_DUMP_DIR,NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS,NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES,NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING,NCCL_INSPECTOR_OTEL_EXPORT \
    bash -c '
        set -euo pipefail
        overrides=()
        [[ -z "$LOGGING" ]] || overrides+=(--logging "$LOGGING")
        [[ -z "$API_PORT" ]] || overrides+=(--api-port "$API_PORT")
        [[ -z "$API_HOST" ]] || overrides+=(--api-host "$API_HOST")
        [[ -z "$API_LINGER" ]] || overrides+=(--api-linger "$API_LINGER")
        exec python /workspace/controller.py --config /config.yml \
            --output /results/suite "${overrides[@]}"
    '
