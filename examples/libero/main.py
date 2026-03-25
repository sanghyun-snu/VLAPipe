import collections
import dataclasses
import logging
import math
import pathlib
import time
import uuid

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc
from examples.pi0_grpc_native.utils.stream_protocol import ndarray_to_proto
from examples.pi0_grpc_native.utils.stream_protocol import proto_to_ndarray
import grpc
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class EvalSummary:
    mode: str
    total_episodes: int
    total_successes: int
    total_requests: int
    request_latencies_s: list[float]
    episode_durations_s: list[float]

    @property
    def success_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return float(self.total_successes) / float(self.total_episodes)

    @property
    def mean_request_latency_s(self) -> float:
        if not self.request_latencies_s:
            return 0.0
        return float(np.mean(self.request_latencies_s))

    @property
    def p95_request_latency_s(self) -> float:
        if not self.request_latencies_s:
            return 0.0
        return float(np.percentile(self.request_latencies_s, 95))

    @property
    def mean_episode_duration_s(self) -> float:
        if not self.episode_durations_s:
            return 0.0
        return float(np.mean(self.episode_durations_s))


def _normalize_action_chunk(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected action shape: {arr.shape}")
    if arr.shape[1] < 7:
        raise ValueError(f"Unexpected action width: {arr.shape[1]}, expected >= 7")
    # Libero environments consume 7-DoF actions.
    arr = arr[:, :7]
    return np.asarray(arr, dtype=np.float32)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 50061
    timeout_s: float = 30.0
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 3  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    execution_mode: str = "org"  # org | v1 | v2
    compare_all_modes: bool = False
    warmup_diffusion_steps: int = 1
    max_inflight_updates: int = 2
    cache_ttl_ms: int = 0
    allow_stale_cache: bool = False
    max_staleness_layers: int = 0
    drop_late_updates: bool = False


def _build_execution_config(args: Args, mode: str) -> pb2.ExecutionConfig:
    cfg = pb2.ExecutionConfig(require_baseline_equivalence=True)
    if mode == "v1":
        cfg.mode = pb2.SUFFIX_EXECUTION_MODE_LAYER_PIPELINE_V1
        cfg.v1.strict_layer_ordering = True
        cfg.v1.warmup_diffusion_steps = args.warmup_diffusion_steps
        cfg.v1.fail_on_missing_layer = True
        return cfg
    if mode == "v2":
        cfg.mode = pb2.SUFFIX_EXECUTION_MODE_ASYNC_CACHE_V2
        cfg.v2.max_inflight_updates = args.max_inflight_updates
        cfg.v2.cache_ttl_ms = args.cache_ttl_ms
        cfg.v2.allow_stale_cache = args.allow_stale_cache
        cfg.v2.max_staleness_layers = args.max_staleness_layers
        cfg.v2.drop_late_updates = args.drop_late_updates
        return cfg
    raise ValueError(f"Unsupported execution_mode for pipeline request: {mode}")


def _infer_actions(stub: pb2_grpc.SuffixServiceStub, request: pb2.EvalRequest, args: Args, mode: str) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    if mode == "org":
        response = stub.Evaluate(request, timeout=args.timeout_s)
    elif mode in ("v1", "v2"):
        pipeline_request = pb2.EvaluatePipelineRequest(
            eval_request=request,
            execution=_build_execution_config(args, mode),
        )
        response = stub.EvaluateLayerPipeline(pipeline_request, timeout=args.timeout_s)
    else:
        raise ValueError(f"Unsupported execution_mode: {mode}. Expected one of: org, v1, v2")
    latency_s = time.perf_counter() - t0
    return _normalize_action_chunk(proto_to_ndarray(response.actions)), latency_s


def eval_libero(args: Args, *, mode: str) -> EvalSummary:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = pb2_grpc.SuffixServiceStub(channel)
    request_latencies_s: list[float] = []
    episode_durations_s: list[float] = []

    try:
        # Start evaluation
        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description}")
                episode_start_t = time.perf_counter()

                # Reset environment
                env.reset()
                action_plan = collections.deque()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []

                logging.info(f"Starting episode {task_episodes+1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        if not action_plan:
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(task_description),
                            }
                            request = pb2.EvalRequest(
                                request_id=str(uuid.uuid4()),
                                policy_type=pb2.POLICY_TYPE_LIBERO,
                            )
                            request.libero.CopyFrom(
                                pb2.LiberoInput(
                                    state=ndarray_to_proto(np.asarray(element["observation/state"])),
                                    image=ndarray_to_proto(np.asarray(element["observation/image"])),
                                    wrist_image=ndarray_to_proto(np.asarray(element["observation/wrist_image"])),
                                    prompt=str(element.get("prompt", "")),
                                )
                            )
                            action_chunk, request_latency_s = _infer_actions(stub, request, args, mode)
                            request_latencies_s.append(request_latency_s)
                            logging.info(
                                f"[{mode}] request latency={request_latency_s:.4f}s "
                                f"task={task_id} episode={episode_idx} t={t}"
                            )
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.error(f"Caught exception: {e}")
                        break

                task_episodes += 1
                total_episodes += 1
                episode_duration_s = time.perf_counter() - episode_start_t
                episode_durations_s.append(episode_duration_s)

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"[{mode}] episode_duration_s={episode_duration_s:.4f}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
        logging.info(f"Total episodes: {total_episodes}")
    finally:
        channel.close()

    summary = EvalSummary(
        mode=mode,
        total_episodes=total_episodes,
        total_successes=total_successes,
        total_requests=len(request_latencies_s),
        request_latencies_s=request_latencies_s,
        episode_durations_s=episode_durations_s,
    )
    logging.info(
        f"[{mode}] summary success_rate={summary.success_rate:.4f} "
        f"mean_request_latency_s={summary.mean_request_latency_s:.4f} "
        f"p95_request_latency_s={summary.p95_request_latency_s:.4f} "
        f"mean_episode_duration_s={summary.mean_episode_duration_s:.4f}"
    )
    return summary


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    if args.compare_all_modes:
        results = []
        for mode in ("org", "v1", "v2"):
            run_args = dataclasses.replace(args, execution_mode=mode)
            logging.info(f"=== Starting mode={mode} ===")
            results.append(eval_libero(run_args, mode=mode))
        logging.info("=== Mode Comparison (org/v1/v2) ===")
        for result in results:
            logging.info(
                f"{result.mode}: success={result.total_successes}/{result.total_episodes} ({result.success_rate * 100:.1f}%) "
                f"mean_req={result.mean_request_latency_s:.4f}s p95_req={result.p95_request_latency_s:.4f}s "
                f"mean_ep={result.mean_episode_duration_s:.4f}s requests={result.total_requests}"
            )
    else:
        eval_libero(args, mode=args.execution_mode)
