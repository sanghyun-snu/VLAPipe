from __future__ import annotations

import asyncio

from .eval import main_async as eval_main_async
from .eval import build_arg_parser as eval_arg_parser
from .prefix import PrefixServer
from .suffix import SuffixServer

SMOKE_PREFIX_PORT = 50072
SMOKE_SUFFIX_PORT = 50073


async def smoke() -> None:
    prefix_server = PrefixServer(host="127.0.0.1", port=SMOKE_PREFIX_PORT)
    suffix_server = SuffixServer(
        host="127.0.0.1",
        port=SMOKE_SUFFIX_PORT,
        prefix_host="127.0.0.1",
        prefix_port=SMOKE_PREFIX_PORT,
    )
    await prefix_server.start()
    await suffix_server.start()

    try:
        args = eval_arg_parser().parse_args(
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(SMOKE_SUFFIX_PORT),
                "--policy",
                "droid",
                "--num-layers",
                "4",
                "--hidden-size",
                "32",
                "--compute-delay-s",
                "0.02",
                "--poll-interval-s",
                "0.01",
                "--seed",
                "42",
            ]
        )
        await eval_main_async(args)
    finally:
        await suffix_server.stop()
        await prefix_server.stop()


def main() -> None:
    asyncio.run(smoke())


if __name__ == "__main__":
    main()
