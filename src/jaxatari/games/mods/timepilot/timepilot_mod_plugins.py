
"""Paper-eval robustness mods for Time Pilot (turn-family parallel to Asteroids)."""
from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_timepilot import TimePilotState


class SpawnAxisBiasMod(JaxAtariPostStepModPlugin):
    """S: force enemy headings to cardinal L/R (rotations 2=left, 6=right)."""

    name = "spawn_axis_bias"

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: TimePilotState, new_state: TimePilotState) -> TimePilotState:
        del prev_state
        enemies = new_state.enemy_states
        rot = enemies[:, 2].astype(jnp.int32)
        horiz = jnp.where((rot % 8) < 4, jnp.int32(2), jnp.int32(6))
        enemies = enemies.at[:, 2].set(horiz.astype(enemies.dtype))
        return new_state.replace(enemy_states=enemies)


class PlaneInertiaMod(JaxAtariInternalModPlugin):
    """D_p: heavy world-scroll inertia (±1 vs default ±4) — clearly felt."""

    name = "plane_inertia"
    constants_overrides = {
        "PLAYER_SPEED_PER_ROTATION": jnp.array(
            [
                (0, -1),
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
                (1, 0),
                (1, -1),
            ],
            dtype=jnp.int32,
        ),
    }


class FasterEnemiesMod(JaxAtariPostStepModPlugin):
    """D_t: apply an extra enemy step along facing."""

    name = "faster_enemies"

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: TimePilotState, new_state: TimePilotState) -> TimePilotState:
        del prev_state
        level = self._env._get_level_constants(new_state.level)
        enemies = new_state.enemy_states

        def _boost(e):
            x, y, rot, active = e[0], e[1], e[2].astype(jnp.int32), e[3]
            dx = level.enemy_speed_per_rotation[rot][0]
            dy = level.enemy_speed_per_rotation[rot][1]
            nx = jnp.where(active != 0, x + dx, x)
            ny = jnp.where(active != 0, y + dy, y)
            return jnp.array([nx, ny, e[2], active], dtype=e.dtype)

        enemies = jax.vmap(_boost)(enemies)
        return new_state.replace(enemy_states=enemies)


class FrozenBaitMod(JaxAtariPostStepModPlugin):
    """A: one frozen bait at a time on a cardinal axis; vary distance; no shots."""

    name = "frozen_bait"

    # Cardinal dirs (L, R, U, D) with missile spawn offsets from player_missile_step
    # so L/R shots share the enemy Y band (py+6 put enemy below the missile).
    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    MISSILE_ORIGINS = ((0, 4), (7, 4), (3, 0), (3, 13))  # L, R, U, D
    DISTANCES = (22, 30, 38, 48, 58)  # never overlaps the player ship
    ENEMY_W = 8
    ENEMY_H = 7  # level-1 size at rotation 2

    def _spawn_one(self, rng, dtype):
        px = jnp.int32(self._env.consts.PLAYER_X)
        py = jnp.int32(self._env.consts.PLAYER_Y)
        dirs = jnp.array(self.DIRS, dtype=jnp.int32)
        origins = jnp.array(self.MISSILE_ORIGINS, dtype=jnp.int32)
        dists = jnp.array(self.DISTANCES, dtype=jnp.int32)
        rng, k_dir, k_dist = jax.random.split(rng, 3)
        di = jax.random.randint(k_dir, (), 0, dirs.shape[0])
        dist = dists[jax.random.randint(k_dist, (), 0, dists.shape[0])]
        d = dirs[di]
        o = origins[di]
        cx = px + o[0] + d[0] * dist
        cy = py + o[1] + d[1] * dist
        bait = jnp.array(
            [
                cx - jnp.int32(self.ENEMY_W // 2),
                cy - jnp.int32(self.ENEMY_H // 2),
                jnp.int32(2),
                jnp.int32(1),
            ],
            dtype=dtype,
        )
        return bait, rng

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: TimePilotState):
        n = state.enemy_states.shape[0]
        bait, rng = self._spawn_one(state.rng_key, state.enemy_states.dtype)
        enemies = jnp.zeros((n, 4), dtype=state.enemy_states.dtype)
        enemies = enemies.at[0].set(bait)
        missiles = state.enemy_missile_states.at[:, 3].set(0)
        state = state.replace(
            enemy_states=enemies,
            enemy_missile_states=missiles,
            enemy_shot_timer=jnp.int32(10_000),
            rng_key=rng,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: TimePilotState, new_state: TimePilotState) -> TimePilotState:
        prev_e = prev_state.enemy_states
        new_e = new_state.enemy_states
        n = new_e.shape[0]
        px = jnp.int32(self._env.consts.PLAYER_X) + jnp.int32(3)
        py = jnp.int32(self._env.consts.PLAYER_Y) + jnp.int32(6)
        max_d = jnp.int32(max(self.DISTANCES) + 8)

        prev_near = (jnp.abs(prev_e[0, 0] - px) <= max_d) & (
            jnp.abs(prev_e[0, 1] - py) <= max_d
        )
        still = (prev_e[0, 3] != 0) & (new_e[0, 3] != 0) & prev_near

        # TimePilot often zeroes rng_key mid-step — seed from prev + step_counter.
        rng = jax.random.fold_in(prev_state.rng_key, new_state.step_counter)

        def _keep(_):
            return prev_e[0].at[3].set(jnp.int32(1)), rng

        def _new(_):
            return self._spawn_one(rng, new_e.dtype)

        slot0, rng = jax.lax.cond(still, _keep, _new, operand=None)

        enemies = jnp.zeros((n, 4), dtype=new_e.dtype)
        enemies = enemies.at[0].set(slot0)
        missiles = new_state.enemy_missile_states.at[:, 3].set(0)
        return new_state.replace(
            enemy_states=enemies,
            enemy_missile_states=missiles,
            enemy_shot_timer=jnp.int32(10_000),
            rng_key=rng,
        )
