import copy

import numpy as np

try:
    import gymnasium as gymnasium
    from gymnasium.spaces import Box as GymnasiumBox
    from gymnasium.spaces import Dict as GymnasiumDict
except ImportError:
    gymnasium = None
    GymnasiumBox = None
    GymnasiumDict = None

try:
    import gym as legacy_gym
    from gym.spaces import Box as LegacyBox
    from gym.spaces import Dict as LegacyDict
except ImportError:
    legacy_gym = None
    LegacyBox = None
    LegacyDict = None


def _is_box(space):
    return (GymnasiumBox is not None and isinstance(space, GymnasiumBox)) or (
        LegacyBox is not None and isinstance(space, LegacyBox)
    )


def _is_dict(space):
    return (GymnasiumDict is not None and isinstance(space, GymnasiumDict)) or (
        LegacyDict is not None and isinstance(space, LegacyDict)
    )


def _convert_space(obs_space):
    if _is_box(obs_space):
        obs_space = type(obs_space)(obs_space.low, obs_space.high, obs_space.shape)
    elif _is_dict(obs_space):
        for k, v in obs_space.spaces.items():
            obs_space.spaces[k] = _convert_space(v)
        obs_space = type(obs_space)(obs_space.spaces)
    else:
        raise NotImplementedError
    return obs_space


def _convert_obs(obs):
    if isinstance(obs, np.ndarray):
        if obs.dtype == np.float64:
            return obs.astype(np.float32)
        else:
            return obs
    elif isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = _convert_obs(v)
        return obs


def is_legacy_gym_env(env) -> bool:
    if legacy_gym is not None and isinstance(env, legacy_gym.Env):
        if gymnasium is not None and isinstance(env, gymnasium.Env):
            return False
        return True

    obs_space = getattr(env, "observation_space", None)
    return (LegacyBox is not None and isinstance(obs_space, LegacyBox)) or (
        LegacyDict is not None and isinstance(obs_space, LegacyDict)
    )


if gymnasium is not None:
    class GymnasiumSinglePrecision(gymnasium.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)

            obs_space = copy.deepcopy(self.env.observation_space)
            self.observation_space = _convert_space(obs_space)

        def observation(self, observation):
            return _convert_obs(observation)
else:
    GymnasiumSinglePrecision = None


if legacy_gym is not None:
    class LegacySinglePrecision(legacy_gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)

            obs_space = copy.deepcopy(self.env.observation_space)
            self.observation_space = _convert_space(obs_space)

        def observation(self, observation):
            return _convert_obs(observation)
else:
    LegacySinglePrecision = None


def make_single_precision(env):
    if is_legacy_gym_env(env):
        if LegacySinglePrecision is None:
            raise ImportError("Legacy gym support is unavailable.")
        return LegacySinglePrecision(env)

    if GymnasiumSinglePrecision is None:
        raise ImportError("gymnasium is unavailable.")
    return GymnasiumSinglePrecision(env)
