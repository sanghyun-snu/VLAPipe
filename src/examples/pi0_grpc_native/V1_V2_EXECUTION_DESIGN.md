# PI0 gRPC Suffix Execution Design (v1/v2)

This document specifies a concrete, implementation-ready design for:

- v1: layer-pipelined suffix execution with baseline-equivalent behavior
- v2: fully asynchronous cache/update execution with submit/poll/watch/cancel

The goal is to keep `suffix.py` simple and isolate complexity inside strategy modules.

## 1) Proto API Design

Keep existing RPC unchanged for compatibility:

- `SuffixService.Evaluate(EvalRequest) returns (EvalResponse)`

Add new RPCs:

- `EvaluateLayerPipeline(EvaluatePipelineRequest) returns (EvalResponse)` (v1)
- `SubmitEvaluate(SubmitEvaluateRequest) returns (SubmitEvaluateResponse)` (v2)
- `GetEvaluateResult(GetEvaluateResultRequest) returns (GetEvaluateResultResponse)` (v2)
- `WatchEvaluate(WatchEvaluateRequest) returns (stream EvaluateEvent)` (v2)
- `CancelEvaluate(CancelEvaluateRequest) returns (CancelEvaluateResponse)` (v2)

Suggested proto sketch:

```proto
enum SuffixExecutionMode {
  SUFFIX_EXECUTION_MODE_UNSPECIFIED = 0;
  SUFFIX_EXECUTION_MODE_LAYER_PIPELINE_V1 = 1;
  SUFFIX_EXECUTION_MODE_ASYNC_CACHE_V2 = 2;
}

message EvaluatePipelineRequest {
  EvalRequest eval_request = 1;
  ExecutionConfig execution = 2;
}

message ExecutionConfig {
  SuffixExecutionMode mode = 1;
  bool require_baseline_equivalence = 2; // default true for v1 rollout

  oneof mode_config {
    LayerPipelineConfig v1 = 10;
    AsyncCacheConfig v2 = 11;
  }
}

message LayerPipelineConfig {
  bool strict_layer_ordering = 1;
  uint32 warmup_diffusion_steps = 2; // practical default: 1
  bool fail_on_missing_layer = 3;
}

message AsyncCacheConfig {
  uint32 max_inflight_updates = 1;
  uint32 cache_ttl_ms = 2;
  bool allow_stale_cache = 3;
  uint32 max_staleness_layers = 4;
  bool drop_late_updates = 5;
}

enum EvalOperationState {
  EVAL_OP_STATE_UNSPECIFIED = 0;
  EVAL_OP_STATE_QUEUED = 1;
  EVAL_OP_STATE_RUNNING = 2;
  EVAL_OP_STATE_SUCCEEDED = 3;
  EVAL_OP_STATE_FAILED = 4;
  EVAL_OP_STATE_CANCELLED = 5;
}

message SubmitEvaluateRequest {
  EvaluatePipelineRequest request = 1;
  bool wait_for_result = 2;
}

message SubmitEvaluateResponse {
  string operation_id = 1;
  EvalOperationState state = 2;
  EvalResponse result = 3; // set only when completed
  string message = 4;
}

message GetEvaluateResultRequest { string operation_id = 1; }
message GetEvaluateResultResponse {
  string operation_id = 1;
  EvalOperationState state = 2;
  EvalResponse result = 3;
  string error = 4;
}

message WatchEvaluateRequest { string operation_id = 1; }
message EvaluateEvent {
  string operation_id = 1;
  EvalOperationState state = 2;
  int32 layer_idx = 3;
  int64 cache_epoch = 4;
  string event = 5;
  string details = 6;
}

message CancelEvaluateRequest { string operation_id = 1; }
message CancelEvaluateResponse { bool cancelled = 1; }
```

## 2) Internal Execution Architecture

### 2.1 Keep `suffix.py` thin

`suffix.py` should only:

- parse request
- select mode
- delegate to strategy
- map top-level exceptions to gRPC status

### 2.2 Strategy interface

Create a shared strategy contract:

```python
class SuffixExecutionStrategy(Protocol):
    async def run(self, *, request, raw_policy_input, policy_name, context) -> pb2.EvalResponse: ...
```

Recommended module layout:

- `utils/pipeline/execution/base.py`
- `utils/pipeline/execution/v1_layer_pipeline.py`
- `utils/pipeline/execution/v2_async_cache.py`
- `utils/pipeline/execution/registry.py`
- `utils/pipeline/execution/operation_store.py` (v2)
- `utils/pipeline/execution/async_kv_store.py` (v2)

## 3) v1 Procedure (Layer Pipeline + Baseline Equivalence)

Target behavior:

- Overlap only where meaningful: prefix layer arrival with suffix warmup phase
- Keep numerics equivalent to baseline path

Procedure:

1. Start prefix stream (`KVCacheChunk` layer by layer)
2. Validate and ingest into collector
3. If `warmup_diffusion_steps > 0`, run warmup step(s) with layer-ready gating
4. Finalize full cache
5. Run remaining diffusion steps in standard sequential path
6. Return `EvalResponse`

Operational notes:

- practical default: `warmup_diffusion_steps = 1`
- `warmup_diffusion_steps = 0` gives pure baseline behavior
- strict ordering and missing-layer checks enabled by default

## 4) v2 Procedure (Async Cache + Operation Model)

Target behavior:

- Prefix cache updates and suffix compute are independently scheduled
- Client interacts via operation lifecycle, not single blocking RPC

Procedure:

1. `SubmitEvaluate` allocates `operation_id`, state -> `QUEUED`
2. Background job starts:
   - producer task publishes cache epochs
   - consumer task reads epochs according to staleness policy
3. Store status/events/results in operation store
4. `GetEvaluateResult` polls terminal/non-terminal status
5. `WatchEvaluate` streams state transitions and progress
6. `CancelEvaluate` triggers cancellation token propagation

State and safety:

- terminal states: `SUCCEEDED`, `FAILED`, `CANCELLED`
- every operation should have timeout and cleanup policy
- event stream should include `(operation_id, state, cache_epoch)`

## 5) Minimal Integration Plan

Phase A (safe, low risk):

1. Add proto messages/RPCs
2. Keep old `Evaluate` unchanged
3. Implement v1 strategy only
4. Add A/B tests: legacy vs v1(warmup=0/1)

Phase B:

1. Add v2 RPC handlers
2. Add operation store + watch/cancel
3. Start with conservative async policy (no stale cache)

Phase C:

1. Tune v2 staleness/perf
2. Expand observability and regression checks

## 6) Testing and Acceptance Criteria

Required checks:

- legacy `Evaluate` unchanged
- v1 `warmup_diffusion_steps=0` matches baseline output
- v1 `warmup_diffusion_steps=1` no ordering/missing-layer failures
- v2 submit/poll/watch/cancel lifecycle correctness
- timeout/cancel propagate to operation terminal state

Metrics to log:

- first token/layer latency, total latency, queue wait
- warmup step duration vs rest-step duration
- operation queue time, running time, cancel latency

