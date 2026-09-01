#!/bin/bash

# this is all in one wrapper script mainly for container
# No GPU in the container; torch runs the networks on CPU
export BEN_TORCH_DEVICE=cpu

python3 gameserver.py 2>&1 | grep -v "cuda\|cuDNN\|cuBLAS\|cuFFT\|cuInit\|CUDA\|absl::InitializeLog" & # listen on 4443 for websocket

python3 gameapi.py --host 0.0.0.0 2>&1 | grep -v "cuda\|cuDNN\|cuBLAS\|cuFFT\|cuInit\|CUDA\|absl::InitializeLog" & # listen on 8085 for REST API

cd "$(dirname "$0")"/frontend
python3 appserver.py --host 0.0.0.0 &  # listen on 8080 for browser

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?