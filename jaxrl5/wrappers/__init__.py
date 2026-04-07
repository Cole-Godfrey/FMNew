import gymnasium as gymnasium
from gymnasium.wrappers.flatten_observation import FlattenObservation as GymnasiumFlattenObservation

try:
    import gym as legacy_gym
    from gym.wrappers.flatten_observation import FlattenObservation as LegacyFlattenObservation
except ImportError:
    legacy_gym = None
    LegacyFlattenObservation = None

from jaxrl5.wrappers.single_precision import is_legacy_gym_env, make_single_precision


def wrap_gym(env, rescale_actions: bool = True, cost_limit: int = 1):
    env = make_single_precision(env)

    if is_legacy_gym_env(env):
        if legacy_gym is None:
            raise ImportError("Legacy gym support is unavailable.")

        if rescale_actions:
            env = legacy_gym.wrappers.RescaleAction(env, -1, 1)

        if isinstance(env.observation_space, legacy_gym.spaces.Dict):
            env = LegacyFlattenObservation(env)
        env = legacy_gym.wrappers.ClipAction(env)
    else:
        if rescale_actions:
            env = gymnasium.wrappers.RescaleAction(env, -1, 1)

        if isinstance(env.observation_space, gymnasium.spaces.Dict):
            env = GymnasiumFlattenObservation(env)
        env = gymnasium.wrappers.ClipAction(env)

    env.set_target_cost(cost_limit)
    print('env_cost_limit', env.target_cost)
    return env
