import jax
import jax.numpy as jnp
import pytest

import jaxatari
from jaxatari.games.jax_montezumarevenge import JaxMontezumaRevenge
from jaxatari.games.montezuma_revenge.rooms import load_room
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
)


def test_montezuma_observation_includes_ladders_room_and_types():
    env = JaxMontezumaRevenge()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = load_room(12, state, env.consts)
    obs = env._get_observation(state)

    assert int(obs.room_id) == 12
    assert int(jnp.sum(obs.ladders.active)) >= 1
    assert int(obs.ladders.x[0]) == int(state.ladders_x[0])
    assert int(obs.items.visual_id[0]) == int(state.items_type[0])
    assert int(obs.enemies.visual_id[0]) == int(state.enemies_type[0])
    assert jnp.allclose(obs.inventory, state.inventory)


def test_montezuma_inactive_entities_are_zeroed():
    env = JaxMontezumaRevenge()
    _, state = env.reset(jax.random.PRNGKey(0))
    obs = env._get_observation(state)

    inactive_enemy = int(jnp.where(obs.enemies.active == 0, size=1, fill_value=0)[0][0])
    assert int(obs.enemies.x[inactive_enemy]) == 0
    assert int(obs.enemies.y[inactive_enemy]) == 0


def test_montezuma_object_centric_wrapper_accepts_observation_space():
    env = jaxatari.make("montezumarevenge")
    env = AtariWrapper(env, episodic_life=True, first_fire=True, noop_max=30)
    env = ObjectCentricWrapper(env, frame_stack_size=4, frame_skip=4, clip_reward=False)
    env = NormalizeObservationWrapper(env)
    env = FlattenObservationWrapper(env)

    obs, state = env.reset(jax.random.PRNGKey(0))
    assert obs.shape == (env.observation_space().shape[0],)
    assert jnp.isfinite(obs).all()

    obs2, state, _, _, _, _ = env.step(state, 3)
    assert obs2.shape == obs.shape
    assert float(jnp.mean(jnp.abs(obs2 - obs))) > 0.0
