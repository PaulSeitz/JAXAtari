"""Smoke + behavior checks for MODS_PLAN §5 Seaquest / Phoenix additions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxatari.core import make
from jaxatari.environment import JAXAtariAction as Action


def _base_env(env):
    while hasattr(env, "_env"):
        env = env._env
    return env


@pytest.mark.parametrize(
    "mod_key",
    [
        "mirror_divers",
        "mirror_world",
        "inverted_controls",
        "momentum",
        "sparse_world",
        "tiny_divers",
        "dynamic_lane_drift",
        "continuous_random_spawns",
        "ocean_currents",
        "erratic_enemies",
        "micro_swarm",
        "sticky_water",
        "ghost_sharks",
        "slippery_turn",
        "lane_scramble",
    ],
)
def test_seaquest_new_mod_reset_and_step(mod_key):
    env = make("seaquest", mods=[mod_key])
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    obs2, state2, reward, done, info = env.step(state, jnp.int32(0))
    assert obs is not None
    assert obs2 is not None
    _ = float(reward)
    _ = bool(done)


def test_tiny_divers_shrinks_hitbox():
    env = _base_env(make("seaquest", mods=["tiny_divers"]))
    assert tuple(int(x) for x in env.consts.DIVER_SIZE) == (4, 5)


def test_ocean_currents_drifts_player_left():
    env = make("seaquest", mods=["ocean_currents"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(6))
    state = state.replace(
        player_x=jnp.int32(100),
        player_y=jnp.int32(100),  # underwater
        step_counter=jnp.int32(0),
    )
    from jaxatari.games.mods.seaquest.seaquest_mod_plugins import OceanCurrentsMod

    plugin = OceanCurrentsMod()
    plugin._env = core
    # Force a drift tick (step_counter % 3 == 0).
    drifted = plugin.run(state, state.replace(step_counter=jnp.int32(3)))
    assert int(drifted.player_x) == int(state.player_x) - 1


def test_micro_swarm_shrinks_enemy_hitboxes():
    env = _base_env(make("seaquest", mods=["micro_swarm"]))
    assert tuple(int(x) for x in env.consts.SHARK_SIZE) == (3, 3)
    assert tuple(int(x) for x in env.consts.ENEMY_SUB_SIZE) == (3, 3)


def test_continuous_random_spawns_holds_y():
    env = make("seaquest", mods=["continuous_random_spawns"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(7))
    ys = [int(y) for y in core.consts.SPAWN_POSITIONS_Y]
    # Seed a live shark with a non-lane Y; next step must preserve it (not snap).
    held_y = jnp.int32((ys[0] + ys[-1]) // 2)
    sharks = jnp.zeros_like(state.shark_positions).at[0].set(
        jnp.array([40, held_y, 1], dtype=state.shark_positions.dtype)
    )
    state = state.replace(
        shark_positions=sharks,
        sub_positions=jnp.zeros_like(state.sub_positions),
        diver_positions=jnp.zeros_like(state.diver_positions),
        player_x=jnp.int32(100),
        player_y=jnp.int32(ys[-1]),
    )
    _, state2, *_ = env.step(state, jnp.int32(0))
    assert int(state2.shark_positions[0, 1]) == int(held_y)
    assert 50 <= int(held_y) <= 160


def test_random_spawns_collision_matches_visual_y():
    """Hitboxes must use the random Y — not the stock lane — or remote deaths occur."""
    env = make("seaquest", mods=["random_spawns"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(11))
    ys = [int(y) for y in core.consts.SPAWN_POSITIONS_Y]
    visual_y = jnp.int32(ys[2])
    # Shark visually at deep lane; player sitting on stock lane-0 Y at same X.
    sharks = jnp.zeros_like(state.shark_positions).at[0].set(
        jnp.array([80, visual_y, 1], dtype=state.shark_positions.dtype)
    )
    state = state.replace(
        shark_positions=sharks,
        sub_positions=jnp.zeros_like(state.sub_positions),
        diver_positions=jnp.zeros_like(state.diver_positions),
        enemy_missile_positions=jnp.zeros_like(state.enemy_missile_positions),
        surface_sub_position=jnp.zeros(3, dtype=state.surface_sub_position.dtype),
        player_x=jnp.int32(80),
        player_y=jnp.int32(ys[0]),
        lives=jnp.int32(3),
        death_counter=jnp.int32(0),
    )
    lives_before = int(state.lives)
    _, state2, *_ = env.step(state, jnp.int32(0))
    assert int(state2.lives) == lives_before
    assert int(state2.death_counter) == 0
    # Player overlapping the *visual* shark must still die.
    state_hit = state.replace(player_y=visual_y, lives=jnp.int32(3))
    _, state3, *_ = env.step(state_hit, jnp.int32(0))
    assert int(state3.death_counter) > 0 or int(state3.lives) < 3


def test_dynamic_lane_drift_moves_lanes():
    env = make("seaquest", mods=["dynamic_lane_drift"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(8))
    ys = [int(y) for y in core.consts.SPAWN_POSITIONS_Y]
    sharks = jnp.zeros_like(state.shark_positions).at[0].set(
        jnp.array([40, ys[0], 1], dtype=state.shark_positions.dtype)
    )
    # Step 50 → non-zero sinusoid offset on lane 0 with the chosen freq/phase.
    state = state.replace(
        shark_positions=sharks,
        sub_positions=jnp.zeros_like(state.sub_positions),
        diver_positions=jnp.zeros_like(state.diver_positions),
        step_counter=jnp.int32(50),
        player_x=jnp.int32(120),
        player_y=jnp.int32(ys[-1]),
    )
    _, state2, *_ = env.step(state, jnp.int32(0))
    assert int(state2.shark_positions[0, 1]) != ys[0]

def test_erratic_enemies_can_burst_or_pause():
    env = make("seaquest", mods=["erratic_enemies"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(9))
    from jaxatari.games.mods.seaquest.seaquest_mod_plugins import ErraticEnemiesMod

    plugin = ErraticEnemiesMod()
    plugin._env = core
    sharks = state.shark_positions.at[0].set(
        jnp.array([80, 71, 1], dtype=state.shark_positions.dtype)
    )
    # beat = (step + 0) % 16 >= 12 → burst at step 12
    state = state.replace(shark_positions=sharks, step_counter=jnp.int32(12))
    out = plugin.run(state, state)
    assert int(out.shark_positions[0, 0]) == 82  # +2 burst


def test_inverted_controls_swaps_left_right():
    env = _base_env(make("seaquest", mods=["inverted_controls"]))
    _, state = env.reset(jax.random.PRNGKey(1))
    state = state.replace(player_x=jnp.int32(80), player_y=jnp.int32(100), player_direction=jnp.int32(1))
    x_right, _, _ = env.player_step(state, jnp.int32(Action.RIGHT))
    x_left, _, _ = env.player_step(state, jnp.int32(Action.LEFT))
    # RIGHT is remapped to LEFT → x decreases; LEFT → RIGHT → x increases.
    assert int(x_right) < int(state.player_x)
    assert int(x_left) > int(state.player_x)


def test_momentum_coasts_without_input():
    env = _base_env(make("seaquest", mods=["momentum"]))
    _, state = env.reset(jax.random.PRNGKey(2))
    state = state.replace(
        player_x=jnp.int32(80),
        player_y=jnp.int32(100),
        player_direction=jnp.int32(1),
    )
    x, _, _ = env.player_step(state, jnp.int32(Action.NOOP))
    assert int(x) == int(state.player_x) + 1


def test_mirror_world_flips_observation_x():
    vanilla = _base_env(make("seaquest"))
    mirrored = _base_env(make("seaquest", mods=["mirror_world"]))
    key = jax.random.PRNGKey(3)
    _, v_state = vanilla.reset(key)
    _, m_state = mirrored.reset(key)
    # Same underlying state coords; observation x should be mirrored.
    v_obs = vanilla._get_observation(v_state)
    m_obs = mirrored._get_observation(m_state)
    w = int(v_obs.player.width)
    assert int(m_obs.player.x) == 160 - int(v_obs.player.x) - w


def test_sparse_world_pushes_nearby_targets():
    env = make("seaquest", mods=["sparse_world"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(4))
    # Place a shark 10px from the player.
    sharks = state.shark_positions.at[0].set(
        jnp.array([state.player_x + 10, state.player_y, 1], dtype=state.shark_positions.dtype)
    )
    state = state.replace(shark_positions=sharks)
    # Post-step mod runs through wrapper step; call plugin run directly.
    from jaxatari.games.mods.seaquest.seaquest_mod_plugins import SparseWorldMod

    plugin = SparseWorldMod()
    plugin._env = core
    new_state = plugin.run(state, state)
    shark = new_state.shark_positions[0]
    dx = float(shark[0] - new_state.player_x)
    dy = float(shark[1] - new_state.player_y)
    dist = (dx * dx + dy * dy) ** 0.5
    assert dist >= 50.5


def test_phoenix_formation_reshuffle_loads():
    env = make("phoenix", mods=["formation_reshuffle"])
    obs, state = env.reset(jax.random.PRNGKey(5))
    obs2, state2, reward, done, info = env.step(state, jnp.int32(0))
    core = _base_env(env)
    # Formation 0 slot 0 should differ from vanilla 66.
    assert float(core.consts.ENEMY_POSITIONS_X[0, 0]) != 66.0
    _ = float(reward)


def test_sticky_water_slows_underwater_move():
    env = _base_env(make("seaquest", mods=["sticky_water"]))
    _, state = env.reset(jax.random.PRNGKey(10))
    state = state.replace(
        player_x=jnp.int32(80),
        player_y=jnp.int32(100),
        player_direction=jnp.int32(1),
        step_counter=jnp.int32(1),  # odd → blocked underwater
    )
    x, y, _ = env.player_step(state, jnp.int32(Action.RIGHT))
    assert int(x) == int(state.player_x)
    assert int(y) == int(state.player_y)
    state_even = state.replace(step_counter=jnp.int32(2))
    x2, _, _ = env.player_step(state_even, jnp.int32(Action.RIGHT))
    assert int(x2) == int(state.player_x) + 1


def test_slippery_turn_lags_facing():
    env = _base_env(make("seaquest", mods=["slippery_turn"]))
    _, state = env.reset(jax.random.PRNGKey(11))
    state = state.replace(
        player_x=jnp.int32(80),
        player_y=jnp.int32(100),
        player_direction=jnp.int32(1),
        step_counter=jnp.int32(3),  # not a turn frame (period 8)
    )
    _, _, facing = env.player_step(state, jnp.int32(Action.LEFT))
    assert int(facing) == 1  # still facing right
    state_turn = state.replace(step_counter=jnp.int32(8))
    _, _, facing2 = env.player_step(state_turn, jnp.int32(Action.LEFT))
    assert int(facing2) == -1


def test_ghost_sharks_missile_passes_through():
    env = _base_env(make("seaquest", mods=["ghost_sharks"]))
    _, state = env.reset(jax.random.PRNGKey(12))
    sharks = state.shark_positions.at[0].set(
        jnp.array([50, 71, 1], dtype=state.shark_positions.dtype)
    )
    missile = jnp.array([50, 71, 1], dtype=state.player_missile_position.dtype)
    (
        new_missile,
        new_sharks,
        _subs,
        _score,
        _spawn,
        _rng,
    ) = env.check_missile_collisions(
        missile,
        sharks,
        state.sub_positions,
        state.score,
        state.successful_rescues,
        state.spawn_state,
        state.rng_key,
    )
    assert int(new_sharks[0, 2]) != 0  # shark survives
    assert int(new_missile[2]) != 0  # missile not consumed by ghost


def test_lane_scramble_remaps_y():
    env = make("seaquest", mods=["lane_scramble"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(13))
    ys = [int(y) for y in core.consts.SPAWN_POSITIONS_Y]
    # Off-screen: on-screen entities hold Y across epochs, but off-screen ones remap.
    spawn_x = int(core.consts.ENEMY_SPAWN_X_FROM_LEFT)
    sharks = jnp.zeros_like(state.shark_positions).at[0].set(
        jnp.array([spawn_x, ys[0], 1], dtype=state.shark_positions.dtype)
    )
    # Epoch 1 → perm (1,0,3,2): lane 0 → band of lane 1
    state = state.replace(
        shark_positions=sharks,
        sub_positions=jnp.zeros_like(state.sub_positions),
        diver_positions=jnp.zeros_like(state.diver_positions),
        step_counter=jnp.int32(90),
        player_x=jnp.int32(120),
        player_y=jnp.int32(ys[-1]),
    )
    _, state2, *_ = env.step(state, jnp.int32(0))
    # Remap lands on lane-1 band; shark bob may add ± a few px.
    assert abs(int(state2.shark_positions[0, 1]) - ys[1]) <= 4


def test_lane_scramble_does_not_teleport_onto_player():
    env = make("seaquest", mods=["lane_scramble"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(14))
    ys = [int(y) for y in core.consts.SPAWN_POSITIONS_Y]
    # Epoch 2 → perm (2,3,0,1): lane 0 → band of lane 2. Player sits there.
    # Prev frame kept an on-screen shark on the lane-1 band (epoch-1 mapping).
    # Mid-screen reshuffle must hold that Y — reject-vs-stale-player is not enough.
    sharks_prev = jnp.zeros_like(state.shark_positions).at[0].set(
        jnp.array([80, ys[1], 1], dtype=state.shark_positions.dtype)
    )
    state = state.replace(
        shark_positions=sharks_prev,
        sub_positions=jnp.zeros_like(state.sub_positions),
        diver_positions=jnp.zeros_like(state.diver_positions),
        enemy_missile_positions=jnp.zeros_like(state.enemy_missile_positions),
        surface_sub_position=jnp.zeros(3, dtype=state.surface_sub_position.dtype),
        player_x=jnp.int32(80),
        player_y=jnp.int32(ys[2]),
        step_counter=jnp.int32(180),
        lives=jnp.int32(3),
        death_counter=jnp.int32(0),
    )
    _, state2, *_ = env.step(state, jnp.int32(0))
    shark = state2.shark_positions[0]
    assert int(state2.death_counter) == 0
    assert int(state2.lives) == 3
    # Held on prior on-screen band (bob may nudge a few px).
    assert abs(int(shark[1]) - ys[1]) <= 4
    overlaps = bool(
        core.check_collision_single(
            jnp.array([state2.player_x, state2.player_y]),
            core.consts.PLAYER_SIZE,
            shark[:2],
            core.consts.SHARK_SIZE,
        )
    )
    assert not overlaps


def test_lane_scramble_holds_onscreen_y_across_epoch_even_if_player_moved():
    """Spawn_step only sees pre-move player pose; hold must not depend on it."""
    env = make("seaquest", mods=["lane_scramble"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(15))
    ys = [int(y) for y in core.consts.SPAWN_POSITIONS_Y]
    # On-screen shark on lane-3 band; epoch flip wants lane 2. Player starts
    # clear but will move onto the remapped band this frame (DOWN).
    sharks_prev = jnp.zeros_like(state.shark_positions).at[0].set(
        jnp.array([76, ys[3], 1], dtype=state.shark_positions.dtype)
    )
    state = state.replace(
        shark_positions=sharks_prev,
        sub_positions=jnp.zeros_like(state.sub_positions),
        diver_positions=jnp.zeros_like(state.diver_positions),
        enemy_missile_positions=jnp.zeros_like(state.enemy_missile_positions),
        surface_sub_position=jnp.zeros(3, dtype=state.surface_sub_position.dtype),
        player_x=jnp.int32(76),
        player_y=jnp.int32(ys[2] - 6),  # above lane-2 band; DOWN closes the gap
        player_direction=jnp.int32(1),
        step_counter=jnp.int32(180),  # epoch 2 → perm (2,3,0,1): lane0→lane2
        lives=jnp.int32(3),
        death_counter=jnp.int32(0),
        oxygen=jnp.int32(60),
    )
    _, state2, *_ = env.step(state, jnp.int32(Action.DOWN))
    shark = state2.shark_positions[0]
    assert int(state2.death_counter) == 0
    assert abs(int(shark[1]) - ys[3]) <= 4


def test_swap_diver_shark_roles_collects_shark_kills_on_diver():
    """Physics swap: shark contact rescues; diver contact starts death; torpedo clears divers."""
    env = make("seaquest", mods=["swap_diver_shark_roles"])
    core = _base_env(env)
    _, state = env.reset(jax.random.PRNGKey(11))

    # Place a live shark on the player — should collect, not die.
    shark = jnp.array(
        [int(state.player_x), int(state.player_y), 1],
        dtype=state.shark_positions.dtype,
    )
    state_shark = state.replace(
        shark_positions=state.shark_positions.at[0].set(shark),
        divers_collected=jnp.int32(0),
        death_counter=jnp.int32(0),
    )
    from jaxatari.games.mods.seaquest.seaquest_mod_plugins import (
        CollectSharksOnContactMod,
        LethalDiversMod,
        ShootableDiversMod,
    )

    collect = CollectSharksOnContactMod()
    collect._env = core
    after_shark = collect.run(state_shark, state_shark)
    assert int(after_shark.divers_collected) == 1
    assert int(after_shark.shark_positions[0, 2]) == 0
    assert int(after_shark.death_counter) == 0

    # Place a live diver on the player — should trigger death animation.
    diver = jnp.array(
        [int(state.player_x), int(state.player_y), 1],
        dtype=state.diver_positions.dtype,
    )
    state_diver = state.replace(
        diver_positions=state.diver_positions.at[0].set(diver),
        death_counter=jnp.int32(0),
    )
    lethal = LethalDiversMod()
    lethal._env = core
    after_diver = lethal.run(state_diver, state_diver)
    assert int(after_diver.death_counter) == 90

    # Torpedo on a diver — should clear the slot and award kill points.
    state_shot = state.replace(
        diver_positions=state.diver_positions.at[0].set(diver),
        player_missile_position=jnp.array(
            [int(state.player_x), int(state.player_y), 1],
            dtype=state.player_missile_position.dtype,
        ),
        score=jnp.int32(0),
        successful_rescues=jnp.int32(0),
    )
    shoot = ShootableDiversMod()
    shoot._env = core
    after_shot = shoot.run(state_shot, state_shot)
    assert int(after_shot.diver_positions[0, 2]) == 0
    assert int(after_shot.player_missile_position[2]) == 0
    assert int(after_shot.score) == int(core.calculate_kill_points(jnp.int32(0)))


def test_swap_diver_shark_roles_reset_and_step():
    env = make("seaquest", mods=["swap_diver_shark_roles"])
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    obs2, state2, reward, done, info = env.step(state, jnp.int32(0))
    assert obs is not None
    assert obs2 is not None
    _ = float(reward)
    _ = bool(done)
