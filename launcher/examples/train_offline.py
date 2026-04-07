import os
import sys
sys.path.append('.')
import random
import numpy as np
from absl import app, flags
import datetime
import yaml
from ml_collections import config_flags, ConfigDict
import wandb
from tqdm.auto import trange  # noqa
import gymnasium as gym
from env.env_list import env_list
from env.point_robot import PointRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import (
    SafeDiffusion,
    SafeFlowQ,
    SafeFlowQDiffusion,
    SafeFlowQV2,
    SafeFlowQCFM,
    SafeFlowQCFMBudget,
)
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_pr, evaluate_budget
import json


FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_string('project', '', 'project name for wandb')
flags.DEFINE_string('experiment_name', '', 'experiment name for wandb')
flags.DEFINE_string('mode', 'full', 'Training mode: pretrain_critics, full, or finetune')
flags.DEFINE_string('load_model', '', 'Path to pretrained model for finetune mode')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config


def call_main(details):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=details['agent_kwargs'])

    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(
            env,
            critic_type=details['agent_kwargs']['critic_type'],
            data_location=details['dataset_kwargs']['pr_data'],
            balance_cost_binary=details['dataset_kwargs'].get('balance_cost_binary', False),
            balance_seed=details['dataset_kwargs'].get('balance_seed', 0),
        )
    else:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(
            env,
            critic_type=details['agent_kwargs']['critic_type'],
            cost_scale=details['dataset_kwargs']['cost_scale'],
            ratio=details['ratio'],
            balance_cost_binary=details['dataset_kwargs'].get('balance_cost_binary', False),
            balance_seed=details['dataset_kwargs'].get('balance_seed', 0),
        )
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    config_dict = dict(details['agent_kwargs'])
    config_dict['env_max_steps'] = env_max_steps

    model_cls = config_dict.pop("model_cls")
    config_dict.pop("cost_scale", None)
    # Pop experiment naming params (not used by model.create)
    config_dict.pop("sampling_method", None)
    config_dict.pop("actor_objective", None)
    config_dict.pop("extract_method", None)
    config_dict.pop("N", None)
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )

    if details.get('load_model', ''):
        print(f"Loading pretrained model from: {details['load_model']}")
        agent = agent.load(details['load_model'])

    pretrain_steps = details.get('pretrain_steps', 0)
    if details.get('load_model', ''):
        pretrain_steps = 0
        print("Skipping pretrain (load_model is set).")
    save_time = 1

    # Phase 1: Pretrain Vc and Qc
    if pretrain_steps > 0:
        print(f"Phase 1: Pretraining Vc/Qc for {pretrain_steps} steps...")
        for i in trange(pretrain_steps, smoothing=0.1, desc="Pretrain Vc/Qc"):
            sample = ds.sample_jax(256)  # Smaller batch for critic pretraining
            agent, info = agent.update_cost_critics(sample)

            if i % details['log_interval'] == 0:
                wandb.log({f"pretrain/{k}": v for k, v in info.items()}, step=i)

            if i % details['eval_interval'] == 0 and i > 0:
                agent.save(f"./results/{details['group']}/{details['experiment_name']}", f"pretrain_{i}")

        agent.save(f"./results/{details['group']}/{details['experiment_name']}", "pretrain_final")
        print("Phase 1 complete. Vc/Qc pretrained.")

    # Phase 2: Train Actor and Q_flow (with frozen or slow-updating Vc/Qc)
    print(f"Phase 2: Training Actor and Q_flow for {details['max_steps']} steps...")
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])
        agent, info = agent.update(sample)

        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i + pretrain_steps)

        # if i % details['eval_interval'] == 0:
        if i > 0 and i % details['eval_interval'] == 0:
            agent.save(f"./results/{details['group']}/{details['experiment_name']}", save_time)
            save_time += 1
            if details['env_name'] == 'PointRobot':
                eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            elif model_cls == 'SafeFlowQCFMBudget':
                eval_info = evaluate_budget(agent, env, details['eval_episodes'])
            else:
                eval_info = evaluate(agent, env, details['eval_episodes'])
            if details['env_name'] != 'PointRobot':
                eval_info["normalized_return"], eval_info["normalized_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
            wandb.log({f"eval/{k}": v for k, v in eval_info.items()}, step=i + pretrain_steps)


def main(_):
    parameters = FLAGS.config
    if FLAGS.project != '':
        parameters['project'] = FLAGS.project
    parameters['env_name'] = env_list[FLAGS.env_id]
    parameters['ratio'] = FLAGS.ratio
    parameters['mode'] = FLAGS.mode
    parameters['load_model'] = FLAGS.load_model

    if parameters['env_name'] == 'PointRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        parameters['agent_kwargs']['cost_temperature'] = 2
        parameters['agent_kwargs']['reward_temperature'] = 5
        parameters['agent_kwargs']['cost_ub'] = 150
        parameters['agent_kwargs']['N'] = 8

    def build_group_name(p):
        ak = p['agent_kwargs']
        return (
            f"{p['env_name']}_"
            f"{ak.get('sampling_method', 'na')}_"
            f"{ak.get('actor_objective', 'na')}_"
            f"{ak.get('critic_type', 'na')}_"
            f"N{ak.get('N', 'na')}_"
            f"{ak.get('extract_method', 'na')}"
        )

    parameters['group'] = build_group_name(parameters)

    if FLAGS.experiment_name == '':
        parameters['experiment_name'] = (
            parameters['group']
            + '_' + str(datetime.date.today())
            + '_s' + str(parameters['seed'])
            + '_' + str(random.randint(0, 1000))
        )
    else:
        parameters['experiment_name'] = FLAGS.experiment_name

    print(parameters)

    if not os.path.exists(f"./results/{parameters['group']}/{parameters['experiment_name']}"):
        os.makedirs(f"./results/{parameters['group']}/{parameters['experiment_name']}")
    with open(f"./results/{parameters['group']}/{parameters['experiment_name']}/config.json", "w") as f:
        json.dump(to_dict(parameters), f, indent=4)

    call_main(parameters)


if __name__ == '__main__':
    app.run(main)
