from typing import Any


METADRIVE_PREFIX = "OfflineMetadrive-"
POINT_ROBOT_NAME = "PointRobot"


def is_metadrive_env(env_name: str) -> bool:
    return env_name.startswith(METADRIVE_PREFIX)


def is_point_robot_env(env_name: str) -> bool:
    return env_name == POINT_ROBOT_NAME


def make_env(env_name: str) -> Any:
    if is_metadrive_env(env_name):
        return _make_metadrive_env(env_name)

    import gymnasium as gym

    return gym.make(env_name)


def _make_metadrive_env(env_name: str) -> Any:
    try:
        import gym
    except ImportError as exc:
        raise ImportError(
            "MetaDrive environments require the legacy `gym` package. "
            "Install the MetaDrive dependencies from DSRL, for example "
            "`pip install -e .[metadrive]` in a DSRL checkout."
        ) from exc

    try:
        import dsrl.offline_metadrive  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Failed to import `dsrl.offline_metadrive`. "
            "Install `dsrl` with MetaDrive support and "
            "`git+https://github.com/HenryLHH/metadrive_clean.git@main`."
        ) from exc

    try:
        return gym.make(env_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create MetaDrive environment `{env_name}`. "
            "Verify that `dsrl` and the MetaDrive simulator are installed."
        ) from exc
