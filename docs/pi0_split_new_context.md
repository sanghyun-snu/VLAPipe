# PI0 Split New Context (Prefix/Suffix, GPU IPC)

## Goal
- Keep layer-wise split architecture (`prefix` streams per-layer KV, `suffix` consumes KV + denoise).
- Improve latency without breaking parity and keep direct GPU IPC as primary path.

## What Was Patched (1~3)

### 1) Prefix sidecar publish can be skipped in direct mode
- Added config flag: `gpu_ipc_publish_sidecar` (default: `true`).
- Added CLI flags:
  - `--gpu-ipc-publish-sidecar`
  - `--gpu-ipc-publish-direct-only` (disables sidecar publish)
- Wired through:
  - `src/examples/pi0_grpc_native/utils/pipeline/models.py`
  - `src/examples/pi0_grpc_native/utils/pipeline/service_config.py`
  - `src/examples/pi0_grpc_native/prefix.py`
  - `scripts/run_pi0_grpc_prefix.py`
- Behavior:
  - When disabled, prefix still emits valid GPU IPC handles in chunk payload.
  - Prefix no longer calls sidecar `PublishHandle` RPC per tensor.

### 2) Reduce synchronization overhead in publish path
- Added `synchronize` control to `PrefixGpuIpcPublisher.publish_tensor(...)`.
- Added `publish_layer_tensors(...)` that synchronizes once per layer, then exports all tensors with `synchronize=False`.
- Prefix session now uses layer-level publish for:
  - `key`
  - `value`
  - optional `prefix_pad_mask`
- This removes repeated `torch.cuda.synchronize(...)` per tensor.

### 3) Per-layer publish RPC "batching" using concurrent futures
- Added `_publish_many_to_sidecar(...)` in `PrefixGpuIpcPublisher`.
- Uses `stub.PublishHandle.future(...)` for all tensors in the layer, then waits/validates responses.
- Keeps protocol/server compatibility (no proto/C++ change required), while reducing serialized round-trip overhead.

## Expected Runtime Impact
- In direct resolve deployments, use `--gpu-ipc-publish-direct-only` to eliminate sidecar publish RPC overhead.
- For sidecar publish mode, one sync per layer + concurrent publish futures should reduce `emit_overhead_layer`.

## How To Run (Recommended)
- Prefix (direct-only publish):
  - `python3 scripts/run_pi0_grpc_prefix.py --kv-transfer-mode gpu_ipc --gpu-ipc-publish-direct-only ...`
- Suffix (direct resolve):
  - `python3 scripts/run_pi0_grpc_suffix.py --kv-transfer-mode gpu_ipc --gpu-ipc-resolve-mode direct ...`

## Validation Checklist
- Check `prefix_profile*.csv`:
  - `emit_overhead_layer` median should drop compared to old run.
- Run parity command with deterministic noise and compare `max_abs_diff`.
- Confirm no sidecar `PublishHandle` traffic when `--gpu-ipc-publish-direct-only` is enabled.

## Notes
- If suffix mode is `sidecar_only` or `sidecar_fallback`, prefix must publish to sidecar (`--gpu-ipc-publish-sidecar`).
- This patch does not change model math; parity differences are likely from forward-path/mask/cache semantics, not this transport optimization.
