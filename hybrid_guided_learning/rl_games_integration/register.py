"""Register the Genesis Go2 handstand tasks with rl_games' env_configurations
and vecenv registries.

Importing this module is enough — the registration runs as an import side effect.
"""
from rl_games.common import env_configurations, vecenv

from .vecenv import Go2VecEnv


def _make_factory(env_class_name: str):
    """Return an rl_games-shaped factory closure.

    The Genesis env class is resolved lazily inside the closure so registering
    these tasks doesn't import Genesis/tensordict at module load time.
    """
    def factory(config_name, num_actors, **kwargs):
        from hybrid_guided_learning.envs.go2 import SingleHandstand

        env_class = {
            "SingleHandstand": SingleHandstand,
        }[env_class_name]
        return Go2VecEnv(env_class, config_name, num_actors, **kwargs)

    return factory


# vecenv "type tags" — values are factories. The tag is what env_configurations
# entries point at via `vecenv_type`.
vecenv.register(
    "GENESIS_GO2_SINGLE_HANDSTAND", _make_factory("SingleHandstand")
)

# Map each YAML `env_name` to a vecenv type. These strings MUST match
# `env_name:` in cfgs/train/*.yaml.
env_configurations.register(
    "genesis_go2_single_handstand",
    {"vecenv_type": "GENESIS_GO2_SINGLE_HANDSTAND"},
)
