# PI0 Changed-Only Experiment

This experiment isolates only the recently changed prefix emit path:

- `gpu_ipc_publish_sidecar = on` (old behavior)
- `gpu_ipc_publish_sidecar = off` via `--gpu-ipc-publish-direct-only` (new behavior)

Everything else stays fixed:

- same policy/checkpoint/device
- same suffix resolve mode (`direct`)
- same deterministic noise setting
- same request count and timeout

## One-command run

Run from repo root:

```bash
uv run python scripts/benchmark_split_changed_only.py \
  --policy libero \
  --checkpoint-map-json examples/checkpoint_map_pytorch.json \
  --policy-device cuda \
  --num-requests 20 \
  --warmup-runs 1
```

## What it outputs

- `emit_overhead_layer` summary for both trials:
  - count, mean, p50, p95, max (ms)
- end-to-end eval mean latency (ms)
- artifact file paths:
  - prefix profile CSV
  - prefix log
  - suffix log

## Interpretation

- `delta_emit_mean_ms (A-B) > 0` means new direct-only path is faster.
- If emit overhead improves but end-to-end does not, bottleneck likely moved to:
  - suffix denoise
  - resolve/copy path
  - non-overlapped stages

## Latest observed numbers
- A `sidecar_on`:
  - `emit_overhead mean=0.951ms`, `p50=0.804ms`, `p95=2.219ms`, `max=6.961ms`
  - `mean_eval=315.710ms`
- B `direct_only`:
  - `emit_overhead mean=0.194ms`, `p50=0.190ms`, `p95=0.228ms`, `max=4.306ms`
  - `mean_eval=295.343ms`
- Delta:
  - `emit_overhead -0.756ms/layer`
  - `eval -20.367ms/request`

### Practical bound from these numbers
- Emitted layers/request: `378 / 20 = 18.9`
- Remaining direct-only emit cost/request: `0.194 * 18.9 ~= 3.67ms`
- Even with perfect emit overhead (`0ms/layer`), expected floor from emit-only tuning is about:
  - `295.343 - 3.67 ~= 291.67ms`
