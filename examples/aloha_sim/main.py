import dataclasses
import logging
import pathlib
import uuid

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc
from examples.pi0_grpc_native.utils.transport import ndarray_to_proto
from examples.pi0_grpc_native.utils.transport import proto_to_ndarray
import env as _env
from openpi_client import action_chunk_broker
import grpc
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro


def _normalize_action_chunk(actions) -> dict:
    arr = proto_to_ndarray(actions)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected action shape: {arr.shape}")
    return {"actions": arr}


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")

    task: str = "gym_aloha/AlohaTransferCube-v0"
    seed: int = 0

    action_horizon: int = 10

    host: str = "127.0.0.1"
    port: int = 50061
    timeout_s: float = 30.0

    display: bool = False


def main(args: Args) -> None:
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = pb2_grpc.SuffixServiceStub(channel)

    class _AlohaGrpcPolicy:
        def infer(self, obs: dict) -> dict:
            request = pb2.EvalRequest(
                request_id=str(uuid.uuid4()),
                policy_type=pb2.POLICY_TYPE_ALOHA,
            )
            request.aloha.CopyFrom(
                pb2.AlohaInput(
                    state=ndarray_to_proto(obs["state"]),
                    images={name: ndarray_to_proto(image) for name, image in obs["images"].items()},
                    prompt=str(obs.get("prompt", "")),
                )
            )
            response = stub.Evaluate(request, timeout=args.timeout_s)
            return _normalize_action_chunk(response.actions)

        def reset(self) -> None:
            return

    runtime = _runtime.Runtime(
        environment=_env.AlohaSimEnvironment(
            task=args.task,
            seed=args.seed,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_AlohaGrpcPolicy(),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
        ],
        max_hz=50,
    )

    try:
        runtime.run()
    finally:
        channel.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
