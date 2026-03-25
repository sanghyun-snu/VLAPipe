# PI0 GPU IPC Sidecar (C++ Skeleton)

This directory contains the C++ sidecar skeleton for the split prefix/suffix
GPU data path.

Current status:
- Prefix/suffix C++ gRPC sidecar servers are buildable and runnable.
- Python split pipeline defaults to `kv_transfer_mode=gpu_ipc`.
- CUDA IPC publish/resolve is executed in Python transport bridge to remove CPU roundtrip for KV payload.

Planned responsibilities:
- Prefix sidecar: export CUDA IPC handles for layer KV tensors.
- Suffix sidecar: resolve CUDA IPC handles into GPU tensors for denoise.
- Keep gRPC/protobuf as control plane; move tensor payload off protobuf bytes.

Build (skeleton):
```bash
cmake -S native/pi0_sidecar -B build/pi0_sidecar
cmake --build build/pi0_sidecar -j

# run sidecars
./build/pi0_sidecar/prefix_sidecar 127.0.0.1:55062
./build/pi0_sidecar/suffix_sidecar 127.0.0.1:55061
```

