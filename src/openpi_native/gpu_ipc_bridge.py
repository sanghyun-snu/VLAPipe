from __future__ import annotations


class _UnavailableBridge:
    def __init__(self, sidecar_address: str) -> None:
        self._sidecar_address = sidecar_address

    def _raise(self) -> None:
        raise RuntimeError(
            "openpi_native GPU IPC bridge is not built. "
            "Build native/pi0_sidecar and install native Python bindings to use kv_transfer_mode=gpu_ipc."
        )


class PrefixPublisher(_UnavailableBridge):
    def publish_tensor(self, request_id: str, layer_idx: int, name: str, tensor):
        self._raise()


class SuffixResolver(_UnavailableBridge):
    def resolve_tensor(
        self,
        *,
        request_id: str,
        layer_idx: int,
        name: str,
        handle_id: str,
        shape: list[int],
        dtype: str,
        device_index: int,
        nbytes: int,
    ):
        self._raise()

