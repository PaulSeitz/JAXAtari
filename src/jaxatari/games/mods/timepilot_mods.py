
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.timepilot.timepilot_mod_plugins import (
    SpawnAxisBiasMod,
    PlaneInertiaMod,
    FasterEnemiesMod,
    FrozenBaitMod,
)


class TimePilotEnvMod(JaxAtariModController):
    """Game-specific mod controller for Time Pilot."""

    REGISTRY = {
        "spawn_axis_bias": SpawnAxisBiasMod,
        "plane_inertia": PlaneInertiaMod,
        "faster_enemies": FasterEnemiesMod,
        "frozen_bait": FrozenBaitMod,
    }

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
