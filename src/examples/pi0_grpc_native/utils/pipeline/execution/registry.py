from __future__ import annotations

from .v1_layer_pipeline import V1LayerPipelineStrategy
from .v2_async_cache import V2AsyncCacheStrategy
from .v3_kv_polisher import V3KVPolisherStrategy


def make_suffix_execution_strategy(
    *,
    mode: str,
    prefix_client,
    loaded_component,
    config,
    profiler,
):
    if mode == "v1_layer_pipeline":
        return V1LayerPipelineStrategy(
            prefix_client=prefix_client,
            loaded_component=loaded_component,
            config=config,
            profiler=profiler,
        )
    if mode == "v2_async_cache":
        return V2AsyncCacheStrategy(
            prefix_client=prefix_client,
            loaded_component=loaded_component,
            config=config,
            profiler=profiler,
        )
    if mode == "v3_kv_polisher":
        return V3KVPolisherStrategy(
            prefix_client=prefix_client,
            loaded_component=loaded_component,
            config=config,
            profiler=profiler,
        )
    raise ValueError(
        f"Unsupported suffix execution_mode={mode!r}. "
        "Expected one of: 'v1_layer_pipeline', 'v2_async_cache', 'v3_kv_polisher'."
    )
