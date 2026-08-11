"""Asterix mod controller and registry (paper_eval RQ2 core mods)."""

import os

from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.asterix.asterix_mod_plugins import (
    FastEnemiesMod,
    NoEnemiesMod,
    LethalCollectiblesMod,
    TopLanesOnlyMod,
    DenseSpawnsMod,
    RecolorEnemiesMod,
)


class AsterixEnvMod(JaxAtariModController):
    """Game-specific mod controller for Asterix."""

    REGISTRY = {
        "fast_enemies": FastEnemiesMod,
        "no_enemies": NoEnemiesMod,
        "lethal_collectibles": LethalCollectiblesMod,
        "top_lanes_only": TopLanesOnlyMod,
        "dense_spawns": DenseSpawnsMod,
        "recolor_enemies": RecolorEnemiesMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "asterix", "sprites")

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
