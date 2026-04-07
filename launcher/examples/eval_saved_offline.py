import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from absl import app, flags

from env.env_list import env_list  # noqa: F401 (keep import side effects consistency)
from env.factory import is_point_robot_env, make_env
from env.point_robot import PointRobot
from jaxrl5.agents import (
    SafeDiffusion,
    SafeFlowQ,
    SafeFlowQDiffusion,
    SafeFlowQV2,
    SafeFlowQCFM,
    SafeFlowQCFMBudget,
)
from jaxrl5.evaluation import evaluate, evaluate_pr, evaluate_budget
from jaxrl5.wrappers import wrap_gym


FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", None, "Experiment directory containing config.json and model*.pickle")
flags.DEFINE_string("model_file", "", "Optional model file name (e.g. model1.pickle or pretrain_final.pickle)")
flags.DEFINE_integer("eval_episodes", 20, "Number of evaluation episodes")
flags.DEFINE_integer("seed", 0, "Eval env seed")
flags.mark_flag_as_required("model_dir")


def _pick_model_file(model_dir: str, model_file_flag: str) -> str:
    if model_file_flag:
        path = os.path.join(model_dir, model_file_flag)
        if not os.path.exists(path):
            raise FileNotFoundError(f"model_file not found: {path}")
        return path

    files = [f for f in os.listdir(model_dir) if f.endswith(".pickle")]
    if not files:
        raise FileNotFoundError(f"No .pickle files found in {model_dir}")

    numbered = []
    for f in files:
        m = re.search(r"model(\d+)\.pickle$", f)
        if m:
            numbered.append((int(m.group(1)), f))
    if numbered:
        numbered.sort(key=lambda x: x[0])
        return os.path.join(model_dir, numbered[-1][1])

    # Fallback: newest by mtime (handles pretrain_final.pickle, etc.)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    return os.path.join(model_dir, files[-1])


def _load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def main(_):
    cfg = _load_config(FLAGS.model_dir)
    model_path = _pick_model_file(FLAGS.model_dir, FLAGS.model_file)

    env_name = cfg["env_name"]
    is_point_robot = is_point_robot_env(env_name)

    if is_point_robot:
        env = PointRobot(id=0, seed=FLAGS.seed)
    else:
        env = make_env(env_name)
        env_max_steps = getattr(env.spec, "max_episode_steps", None)
        if env_max_steps is None:
            env_max_steps = getattr(env.unwrapped, "_max_episode_steps")
        env = wrap_gym(env, cost_limit=cfg["agent_kwargs"]["cost_limit"])

    if is_point_robot:
        env_max_steps = env._max_episode_steps
    config_dict = dict(cfg["agent_kwargs"])
    model_cls = config_dict.pop("model_cls")

    config_dict["env_max_steps"] = env_max_steps

    # Drop naming-only keys (never passed to create())
    config_dict.pop("actor_objective", None)
    config_dict.pop("extract_method", None)

    # N and sampling_method: valid create() params for SafeFlowQDiffusion/SafeFlowQ,
    # but unknown to SafeDiffusion/SafeFlowQV2 → pop only when not needed
    # N and sampling_method: valid create() params for SafeFlowQDiffusion/SafeFlowQ only
    if model_cls not in ("SafeFlowQDiffusion", "SafeFlowQ"):
        config_dict.pop("sampling_method", None)
        config_dict.pop("N", None)

    # actor_hidden_dims comes as list from JSON, convert to tuple
    if "actor_hidden_dims" in config_dict:
        config_dict["actor_hidden_dims"] = tuple(config_dict["actor_hidden_dims"])

    config_dict.pop("cost_scale", None)

    agent = globals()[model_cls].create(
        cfg["seed"], env.observation_space, env.action_space, **config_dict
    )
    agent = agent.load(model_path)

    if is_point_robot:
        eval_info = evaluate_pr(agent, env, FLAGS.eval_episodes)
    elif model_cls == "SafeFlowQCFMBudget":
        eval_info = evaluate_budget(agent, env, FLAGS.eval_episodes)
    else:
        eval_info = evaluate(agent, env, FLAGS.eval_episodes)
        if hasattr(env, "get_normalized_score"):
            nret, ncost = env.get_normalized_score(eval_info["return"], eval_info["cost"])
            eval_info["normalized_return"] = nret
            eval_info["normalized_cost"] = ncost

    print("========== EVAL RESULT ==========")
    print(f"model_dir: {FLAGS.model_dir}")
    print(f"model_file: {os.path.basename(model_path)}")
    print(f"env_name: {env_name}")
    print(f"episodes: {FLAGS.eval_episodes}")
    for k, v in eval_info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    app.run(main)
