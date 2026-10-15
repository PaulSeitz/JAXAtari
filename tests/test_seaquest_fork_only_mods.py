"""Behavior checks for Seaquest mods present in the robustness fork but not master."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxtari.core import make
from jaxtari.games.mods.seaquest.seaquest_mod_plugins import (
    CollectSharksOnContactMod,
    FasterEnemiesMod,
    FastOxygenDrainMod,
    LethalDiversMod,
    NoEnemyTorpedoesMod,
    PeacefulEnemiesMod,
    ShootableDiversMod,
    SlowOxygenDrainMod,
    VerticalOscillationMod,
)


def _base_env(env):
    while hasattr(env, "_env"):
        env = env._env
    return env


FORK_ONLY_PUBLIC = [
    "change_background_color",
    "collectable_sharks",
    "continuous_random_spawns",
    "dense_spawns",
    "dynamic_lane_drift",
    "erratic_enemies",
    "extra_enemy_type",
    "fast_oxygen_drain",
    "faster_enemies",
    "ghost_sharks",
    "grayscale",
    "inverted_colors",
    "inverted_controls",
    "inverted_lanes",
    "lane_scramble",
    "lethal_divers",
    "micro_swarm",
    "mirror_divers",
    "mirror_world",
    "momentum",
    "new_pattern",
    "no_enemies",
    "no_enemy_torpedoes",
    "no_surfacing_death",
    "ocean_currents",
    "only_sharks",
    "only_submarines",
    "peaceful_enemies",
    "random_spawns",
    "shift_lanes",
    "shootable_divers",
    "slippery_turn",
    "slow_oxygen_drain",
    "slower_enemies",
    "sparse_world",
    "sticky_water",
    "surface_sub_always",
    "swap_diver_enemy_labels",
    "swap_diver_shark_roles",
    "tiny_divers",
    "vertical_oscillation",
]


@pytest.mark.parametrize("mod_key", FORK_ONLY_PUBLIC)
def test_fork_only_mod_reset_and_step(mod_key):
    env = make("seaquest", mods=[mod_key])
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    obs2, state2, reward, done, info = env.step(state, jnp.int32(0))
    assert obs is not None and obs2 is not None
    _ = float(reward)
    _ = bool(done)


def test_change_background_color_is_magenta():
    env = make("seaquest", mods=["change_background_color"])
    core = _base_env(env)
    frame = env.render(env.reset(jax.random.PRNGKey(0))[1])
    # Sample a water pixel away from the left black pillar / HUD.
    pixel = [int(x) for x in frame[100, 80, :3]]
    assert pixel == [180, 0, 180]
    assert tuple(int(x) for x in core.consts.BACKGROUND_COLOR) == (180, 0, 180)


def test_grayscale_makes_rgb_channels_equal():
    env = make("seaquest", mods=["grayscale"])
    _, state = env.reset(jax.random.PRNGKey(1))
    frame = env.render(state)
    assert jnp.allclose(frame[..., 0], frame[..., 1])
    assert jnp.allclose(frame[..., 1], frame[..., 2])


def test_inverted_colors_is_255_minus_base():
    base = make("seaquest")
    inv = make("seaquest", mods=["inverted_colors"])
    key = jax.random.PRNGKey(2)
    _, b_state = base.reset(key)
    _, i_state = inv.reset(key)
    b = base.render(b_state)
    i = inv.render(i_state)
    assert jnp.array_equal(i[..., :3], 255 - b[..., :3])


def test_peaceful_enemies_no_death_on_shark_contact():
    env = make("seaquest", mods=["peaceful_enemies"])
    core = _base_env(env)
    plugin = PeacefulEnemiesMod()
    plugin._env = core
    died, pts = plugin.check_player_collision(
        jnp.int32(80),
        jnp.int32(71),
        jnp.zeros((12, 3), dtype=jnp.int32),
        jnp.array([[80, 71, 1]] + [[0, 0, 0]] * 11, dtype=jnp.int32),
        jnp.zeros(3, dtype=jnp.int32),
        jnp.zeros((4, 3), dtype=jnp.int32),
        jnp.int32(0),
        jnp.int32(0),
    )
    assert not bool(died)
    assert int(pts) == 0


def test_no_enemy_torpedoes_clears_missiles():
    env = make("seaquest", mods=["no_enemy_torpedoes"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(3))
    missiles = state.enemy_missile_positions.at[0].set(
        jnp.array([40, 71, 1], dtype=state.enemy_missile_positions.dtype)
    )
    state = state.replace(enemy_missile_positions=missiles)
    plugin = NoEnemyTorpedoesMod()
    plugin._env = core
    out = plugin.run(state, state)
    assert int(jnp.sum(jnp.abs(out.enemy_missile_positions))) == 0


def test_faster_enemies_boosts_x():
    env = make("seaquest", mods=["faster_enemies"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(4))
    sharks = jnp.zeros_like(state.shark_positions).at[0].set(
        jnp.array([80, 71, 1], dtype=state.shark_positions.dtype)
    )
    state = state.replace(shark_positions=sharks)
    plugin = FasterEnemiesMod()
    plugin._env = core
    out = plugin.run(state, state)
    assert int(out.shark_positions[0, 0]) == 81


def test_slower_enemies_holds_x_on_even_frames():
    from jaxtari.games.mods.seaquest.seaquest_mod_plugins import SlowerEnemiesMod

    env = make("seaquest", mods=["slower_enemies"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(5))
    prev = state.replace(
        shark_positions=jnp.zeros_like(state.shark_positions).at[0].set(
            jnp.array([80, 71, 1], dtype=state.shark_positions.dtype)
        )
    )
    # Base motion advanced X; even step_counter → mod restores prev X.
    moved = prev.replace(
        shark_positions=prev.shark_positions.at[0, 0].set(jnp.int32(81)),
        step_counter=jnp.int32(10),
    )
    plugin = SlowerEnemiesMod()
    plugin._env = core
    out = plugin.run(prev, moved)
    assert int(out.shark_positions[0, 0]) == 80


def test_shift_lanes_moves_spawn_y_down():
    vanilla = _base_env(make("seaquest"))
    shifted = _base_env(make("seaquest", mods=["shift_lanes"]))
    v = [int(y) for y in vanilla.consts.SPAWN_POSITIONS_Y]
    s = [int(y) for y in shifted.consts.SPAWN_POSITIONS_Y]
    assert s == [y + 10 for y in v]


def test_only_submarines_clears_sharks():
    env = make("seaquest", mods=["only_submarines"])
    _, state = env.reset(jax.random.PRNGKey(6))
    sharks = state.shark_positions.at[0].set(
        jnp.array([40, 71, 1], dtype=state.shark_positions.dtype)
    )
    state = state.replace(shark_positions=sharks, sub_positions=jnp.zeros_like(state.sub_positions))
    _, state2, *_ = env.step(state, jnp.int32(0))
    assert int(jnp.max(jnp.abs(state2.shark_positions[:, 2]))) == 0
    assert int(state2.sub_positions[0, 2]) != 0


def test_only_sharks_clears_subs():
    env = make("seaquest", mods=["only_sharks"])
    _, state = env.reset(jax.random.PRNGKey(7))
    subs = state.sub_positions.at[0].set(
        jnp.array([40, 71, 1], dtype=state.sub_positions.dtype)
    )
    state = state.replace(sub_positions=subs)
    _, state2, *_ = env.step(state, jnp.int32(0))
    assert int(jnp.max(jnp.abs(state2.sub_positions[:, 2]))) == 0


def test_vertical_oscillation_moves_y():
    env = make("seaquest", mods=["vertical_oscillation"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(8))
    sharks = jnp.zeros_like(state.shark_positions).at[0].set(
        jnp.array([80, 100, 1], dtype=state.shark_positions.dtype)
    )
    # step 3 → sin offset truncates to ±1 (step 20 truncates to 0).
    state = state.replace(shark_positions=sharks, step_counter=jnp.int32(3))
    plugin = VerticalOscillationMod()
    plugin._env = core
    out = plugin.run(state, state)
    assert int(out.shark_positions[0, 1]) != 100


def test_dense_spawns_shortens_opening_timers():
    vanilla = make("seaquest")
    dense = make("seaquest", mods=["dense_spawns"])
    _, v_state = vanilla.reset(jax.random.PRNGKey(9))
    _, d_state = dense.reset(jax.random.PRNGKey(9))
    assert int(jnp.min(d_state.spawn_state.spawn_timers)) < int(
        jnp.min(v_state.spawn_state.spawn_timers)
    )
    core = _base_env(dense)
    assert int(core.consts.SPAWN_TIMER_RELOAD) == 4


def test_fast_oxygen_drain_extra_tick():
    env = make("seaquest", mods=["fast_oxygen_drain"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(10))
    state = state.replace(
        player_y=jnp.int32(100),
        oxygen=jnp.int32(40),
        step_counter=jnp.int32(16),
    )
    plugin = FastOxygenDrainMod()
    plugin._env = core
    out = plugin.run(state, state)
    assert int(out.oxygen) == 39


def test_slow_oxygen_drain_refills_on_skip():
    env = make("seaquest", mods=["slow_oxygen_drain"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(11))
    state = state.replace(
        player_y=jnp.int32(100),
        oxygen=jnp.int32(40),
        step_counter=jnp.int32(32),  # would_drain and skip_drain
    )
    plugin = SlowOxygenDrainMod()
    plugin._env = core
    out = plugin.run(state, state)
    assert int(out.oxygen) == 41


def test_surface_sub_always_active():
    env = make("seaquest", mods=["surface_sub_always"])
    _, state = env.reset(jax.random.PRNGKey(12))
    _, state2, *_ = env.step(state, jnp.int32(0))
    assert int(state2.surface_sub_position[2]) != 0
    assert int(state2.surface_sub_position[1]) == 45


def test_inverted_lanes_flips_directions():
    vanilla = make("seaquest")
    inv = make("seaquest", mods=["inverted_lanes"])
    key = jax.random.PRNGKey(13)
    _, v_state = vanilla.reset(key)
    _, i_state = inv.reset(key)
    assert jnp.array_equal(
        i_state.spawn_state.lane_directions,
        1 - v_state.spawn_state.lane_directions,
    )


def test_lethal_divers_starts_death():
    env = make("seaquest", mods=["lethal_divers"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(14))
    diver = jnp.array(
        [int(state.player_x), int(state.player_y), 1],
        dtype=state.diver_positions.dtype,
    )
    state = state.replace(
        diver_positions=state.diver_positions.at[0].set(diver),
        death_counter=jnp.int32(0),
    )
    plugin = LethalDiversMod()
    plugin._env = core
    out = plugin.run(state, state)
    assert int(out.death_counter) == 90


def test_shootable_divers_clears_and_scores():
    env = make("seaquest", mods=["shootable_divers"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(15))
    diver = jnp.array([50, 71, 1], dtype=state.diver_positions.dtype)
    state = state.replace(
        diver_positions=state.diver_positions.at[0].set(diver),
        player_missile_position=jnp.array([50, 71, 1], dtype=state.player_missile_position.dtype),
        score=jnp.int32(0),
        successful_rescues=jnp.int32(0),
    )
    plugin = ShootableDiversMod()
    plugin._env = core
    out = plugin.run(state, state)
    assert int(out.diver_positions[0, 2]) == 0
    assert int(out.player_missile_position[2]) == 0
    assert int(out.score) == int(core.calculate_kill_points(jnp.int32(0)))


def test_collectable_sharks_fills_slot():
    env = make("seaquest", mods=["collectable_sharks"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(16))
    shark = jnp.array(
        [int(state.player_x), int(state.player_y), 1],
        dtype=state.shark_positions.dtype,
    )
    state = state.replace(
        shark_positions=state.shark_positions.at[0].set(shark),
        divers_collected=jnp.int32(0),
        death_counter=jnp.int32(0),
    )
    plugin = CollectSharksOnContactMod()
    plugin._env = core
    out = plugin.run(state, state)
    assert int(out.divers_collected) == 1
    assert int(out.shark_positions[0, 2]) == 0
    assert int(out.death_counter) == 0


def test_no_enemies_zeros_threats():
    env = make("seaquest", mods=["no_enemies"])
    _, state = env.reset(jax.random.PRNGKey(17))
    _, state2, *_ = env.step(state, jnp.int32(0))
    assert int(jnp.max(jnp.abs(state2.shark_positions[:, 2]))) == 0
    assert int(jnp.max(jnp.abs(state2.sub_positions[:, 2]))) == 0
    assert int(jnp.max(jnp.abs(state2.enemy_missile_positions[:, 2]))) == 0
    assert int(state2.surface_sub_position[2]) == 0


def test_swap_diver_enemy_labels_swaps_obs_channels():
    vanilla = _base_env(make("seaquest"))
    swapped = _base_env(make("seaquest", mods=["swap_diver_enemy_labels"]))
    key = jax.random.PRNGKey(18)
    _, v_state = vanilla.reset(key)
    # Seed a shark and diver so channels are distinguishable.
    sharks = v_state.shark_positions.at[0].set(
        jnp.array([40, 71, 1], dtype=v_state.shark_positions.dtype)
    )
    divers = v_state.diver_positions.at[0].set(
        jnp.array([90, 100, 1], dtype=v_state.diver_positions.dtype)
    )
    state = v_state.replace(shark_positions=sharks, diver_positions=divers)
    v_obs = vanilla._get_observation(state)
    s_obs = swapped._get_observation(state)
    assert int(s_obs.divers.x[0]) == int(v_obs.enemies.x[0])
    assert int(s_obs.enemies.x[0]) == int(v_obs.divers.x[0])


def test_extra_enemy_type_syncs_timers():
    env = make("seaquest", mods=["extra_enemy_type"])
    _, state = env.reset(jax.random.PRNGKey(19))
    timers = state.spawn_state.spawn_timers
    assert int(jnp.max(timers) - jnp.min(timers)) == 0


def test_mirror_divers_fills_empty_collect_slots():
    env = _base_env(make("seaquest", mods=["mirror_divers"]))
    _, state = env.reset(jax.random.PRNGKey(20))
    # Empty divers, live shark in a decoy lane-front slot.
    sharks = jnp.zeros_like(state.shark_positions).at[0].set(
        jnp.array([55, 71, 1], dtype=state.shark_positions.dtype)
    )
    state = state.replace(
        shark_positions=sharks,
        diver_positions=jnp.zeros_like(state.diver_positions),
    )
    obs = env._get_observation(state)
    assert bool(obs.divers.active[0])
    assert int(obs.divers.x[0]) == 55


def test_new_pattern_alias_matches_extra_enemy_type():
    a = make("seaquest", mods=["new_pattern"])
    b = make("seaquest", mods=["extra_enemy_type"])
    _, sa = a.reset(jax.random.PRNGKey(21))
    _, sb = b.reset(jax.random.PRNGKey(21))
    assert jnp.array_equal(sa.spawn_state.spawn_timers, sb.spawn_state.spawn_timers)
