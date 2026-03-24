from __future__ import annotations

import argparse
import asyncio
import uuid

import torch

from .utils.grpc_cache import SuffixEvalClient
from .utils.stream_protocol import EvalRequest
from .utils.stream_protocol import deserialize_eval_response
from .utils.stream_protocol import serialize_eval_request

DEFAULT_SUFFIX_HOST = "127.0.0.1"
DEFAULT_SUFFIX_PORT = 50061
POLICY_STATE_DIM = {
    "droid": 8,
    "aloha": 14,
    "libero": 8,
}


def make_state(policy_name: str, state_dim: int | None, seed: int) -> torch.Tensor:
    resolved_dim = state_dim if state_dim is not None else POLICY_STATE_DIM.get(policy_name, 8)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(1, resolved_dim, generator=generator, dtype=torch.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eval client for suffix server")
    parser.add_argument("--host", default=DEFAULT_SUFFIX_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SUFFIX_PORT)
    parser.add_argument("--policy", default="droid")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--compute-delay-s", type=float, default=0.05)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser


async def main_async(args: argparse.Namespace) -> None:
    request_id = args.request_id or str(uuid.uuid4())
    state = make_state(args.policy, args.state_dim, seed=args.seed)
    request = EvalRequest(
        request_id=request_id,
        policy_name=args.policy,
        state=state,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        prefix_tokens=args.prefix_tokens,
        suffix_tokens=args.suffix_tokens,
        compute_delay_s=args.compute_delay_s,
        poll_interval_s=args.poll_interval_s,
        seed=args.seed,
    )
    client = SuffixEvalClient(address=f"{args.host}:{args.port}")
    try:
        response_raw = await client.evaluate(serialize_eval_request(request), timeout_s=args.timeout_s)
        response = deserialize_eval_response(response_raw)
        print(f"[eval] request_id={response.request_id}")
        print(f"[eval] message={response.message}")
        print(f"[eval] actions={response.actions.tolist()}")
    finally:
        await client.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
