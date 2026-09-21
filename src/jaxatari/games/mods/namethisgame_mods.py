
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.namethisgame.namethisgame_mod_plugins import (
    ShiftTentacleSpawnXMod,
    SlowerPlayerMod,
    FasterSharkMod,
    _NoSharkParkMod,
    _NoSharkObsMod,
)


class NameThisGameEnvMod(JaxAtariModController):
    """Game-specific mod controller for Name This Game."""

    REGISTRY = {
        "shift_tentacle_spawn_x": ShiftTentacleSpawnXMod,
        "slower_player": SlowerPlayerMod,
        "faster_shark": FasterSharkMod,
        "_no_shark_park": _NoSharkParkMod,
        "_no_shark_obs": _NoSharkObsMod,
        # shark_alive=False is death in NTG — park offscreen + obs inactive instead.
        "no_shark_short_tentacles": ["_no_shark_park", "_no_shark_obs"],
    }

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
