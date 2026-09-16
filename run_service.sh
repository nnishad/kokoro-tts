#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="/home/jugaadu/services/kokoro-tts"
VENV_DIR="${SERVICE_DIR}/.venv"

export CUDA_HOME="/opt/cuda"
export PATH="/opt/cuda/bin:${VENV_DIR}/bin:${PATH}"
export ONNX_PROVIDER="CUDAExecutionProvider"

cd "${SERVICE_DIR}"
exec "${VENV_DIR}/bin/uvicorn" server:app --host 0.0.0.0 --port 8880 --log-level info
