"""Surfacing with 0 divers is lethal in vanilla Seaquest; the mod turns that off."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxatari.core import make
from jaxatari.games.jax_seaquest import JaxSeaquest


def _base_env(env):
    while hasattr(env, "_env"):
        env = env._env
    return env


def _surface_with_zero_divers(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        divers_collected=jnp.int32(0),
        just_surfaced=jnp.int32(0),
        oxygen=jnp.int32(40),
    )
    return env.update_oxygen(
        state, state.player_x, jnp.int32(46), state.player_missile_position
    )


def test_vanilla_zero_diver_surface_loses_life():
    lose_life = _surface_with_zero_divers(JaxSeaquest())[5]
    assert bool(lose_life)


def test_no_surfacing_death_keeps_life_and_nonnegative_divers():
    env = _base_env(make("seaquest", mods=["no_surfacing_death"]))
    (
        _oxygen,
        _x,
        _y,
        _missile,
        _o2_out,
        lose_life,
        divers,
        _reset,
        _surfaced,
        _diff,
    ) = _surface_with_zero_divers(env)
    assert not bool(lose_life)
    assert int(divers) >= 0
