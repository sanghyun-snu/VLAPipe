from __future__ import annotations

import argparse
import asyncio

from .eval import main_async as eval_main_async
from .eval import build_arg_parser as eval_arg_parser
from .prefix import PrefixServer
from .suffix import SuffixServer
from .utils import RuntimePolicyArgs
from .utils import load_prefix_component
from .utils import load_suffix_component

SMOKE_PREFIX_PORT = 50072
SMOKE_SUFFIX_PORT = 50073


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for real prefix/suffix split pipeline.")
    parser.add_argument("--policy", choices=["aloha", "libero"], required=True)
    parser.add_argument("--checkpoint-map-json", default="")
    parser.add_argument("--policy-train-config", default="")
    parser.add_argument("--policy-checkpoint-dir", default="")
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--auto-download-checkpoint", action="store_true")
    parser.add_argument("--force-download-checkpoint", action="store_true")
    parser.add_argument("--auto-convert-checkpoint", action="store_true")
    parser.add_argument("--converted-checkpoint-dir", default="")
    parser.add_argument("--convert-precision", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    return parser


def _runtime_policy_args(args: argparse.Namespace) -> RuntimePolicyArgs:
    return RuntimePolicyArgs(
        policy_train_config=args.policy_train_config,
        policy_checkpoint_dir=args.policy_checkpoint_dir,
        policy_name=args.policy,
        checkpoint_map_json=args.checkpoint_map_json,
        auto_download_checkpoint=args.auto_download_checkpoint,
        force_download_checkpoint=args.force_download_checkpoint,
        policy_device=args.policy_device,
        auto_convert_checkpoint=args.auto_convert_checkpoint,
        converted_checkpoint_dir=args.converted_checkpoint_dir,
        convert_precision=args.convert_precision,
    )


async def smoke(args: argparse.Namespace) -> None:
    runtime_args = _runtime_policy_args(args)
    prefix_component = load_prefix_component(runtime_args)
    suffix_component = load_suffix_component(runtime_args)
    prefix_server = PrefixServer(host="127.0.0.1", port=SMOKE_PREFIX_PORT, loaded_component=prefix_component)
    suffix_server = SuffixServer(
        host="127.0.0.1",
        port=SMOKE_SUFFIX_PORT,
        prefix_host="127.0.0.1",
        prefix_port=SMOKE_PREFIX_PORT,
        loaded_component=suffix_component,
    )
    await prefix_server.start()
    await suffix_server.start()

    try:
        eval_args = eval_arg_parser().parse_args(
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(SMOKE_SUFFIX_PORT),
                "--policy",
                args.policy,
            ]
        )
        await eval_main_async(eval_args)
    finally:
        await suffix_server.stop()
        await prefix_server.stop()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(smoke(args))


if __name__ == "__main__":
    main()
