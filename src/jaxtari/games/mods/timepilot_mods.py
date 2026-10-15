
from jaxtari.modification import JaxtariModController
from jaxtari.games.mods.timepilot.timepilot_mod_plugins import (
    SpawnAxisBiasMod,
    PlaneInertiaMod,
    FasterEnemiesMod,
    FrozenBaitMod,
)


from jaxtari.games.mods.robustness_probes import (
    TimePilotShiftAnchorMod,
    TimePilotNoEnemyFireMod,
)

class TimePilotEnvMod(JaxtariModController):
    """Game-specific mod controller for Time Pilot."""

    REGISTRY = {
        "shift_player_anchor": TimePilotShiftAnchorMod,
        "no_enemy_fire": TimePilotNoEnemyFireMod,

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
