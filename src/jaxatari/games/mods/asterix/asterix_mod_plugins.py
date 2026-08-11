"""Asterix robustness / paper_eval mod plugins."""

from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_asterix import AsterixState
from jaxatari.rendering.jax_rendering_utils import get_base_sprite_dir


def _recolor_lyre(filename: str, new_rgb: tuple[int, int, int]) -> np.ndarray:
    """Load a lyre sprite and replace all opaque pixels with a single RGB color."""
    path = os.path.join(get_base_sprite_dir(), "asterix", filename)
    sprite = np.load(path).copy()
    if sprite.shape[-1] == 3:
        is_transparent = np.all(sprite == 0, axis=-1)
        alpha = np.where(is_transparent, 0, 255).astype(np.uint8)
        sprite = np.concatenate([sprite, alpha[..., None]], axis=-1)
    mask = sprite[..., 3] > 128
    sprite[mask, 0] = new_rgb[0]
    sprite[mask, 1] = new_rgb[1]
    sprite[mask, 2] = new_rgb[2]
    return sprite


class FastEnemiesMod(JaxAtariPostStepModPlugin):
    """Enemies move one extra step of their velocity each frame (≈2× speed)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: AsterixState, new_state: AsterixState) -> AsterixState:
        del prev_state
        paused = (new_state.respawn_timer > 0) | (new_state.character_transition_timer > 0)
        enemies = new_state.enemies
        boosted_x = jnp.where(paused, enemies.x, enemies.x + enemies.vx)
        enemy_w = jnp.float32(8.0)
        screen_w = jnp.float32(self._env.consts.screen_width)
        alive = jnp.where(
            paused,
            enemies.alive,
            (boosted_x >= -enemy_w) & (boosted_x <= screen_w + enemy_w) & enemies.alive,
        )
        return new_state.replace(enemies=enemies.replace(x=boosted_x, alive=alive))


class NoEnemiesMod(JaxAtariPostStepModPlugin):
    """Remove all enemies; lanes only spawn/keep collectibles."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: AsterixState):
        enemies = state.enemies.replace(
            x=jnp.full_like(state.enemies.x, -5.0),
            vx=jnp.zeros_like(state.enemies.vx),
            alive=jnp.zeros_like(state.enemies.alive),
        )
        # Prefer collectible spawns after an item (normally force_enemy_next gates this).
        force = jnp.zeros_like(state.force_enemy_next)
        return obs, state.replace(enemies=enemies, force_enemy_next=force)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: AsterixState, new_state: AsterixState) -> AsterixState:
        del prev_state
        enemies = new_state.enemies.replace(
            x=jnp.full_like(new_state.enemies.x, -5.0),
            vx=jnp.zeros_like(new_state.enemies.vx),
            alive=jnp.zeros_like(new_state.enemies.alive),
        )
        force = jnp.zeros_like(new_state.force_enemy_next)
        return new_state.replace(enemies=enemies, force_enemy_next=force)


class LethalCollectiblesMod(JaxAtariPostStepModPlugin):
    """Collectible contact kills the player instead of awarding points (affordance flip)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: AsterixState, new_state: AsterixState) -> AsterixState:
        collected = new_state.score > prev_state.score
        # Enemy collision already started a respawn this frame — only undo the score.
        enemy_hit_this_frame = (prev_state.respawn_timer == 0) & (
            new_state.respawn_timer == jnp.int32(self._env.consts.respawn_frames)
        )
        should_die = collected & (~enemy_hit_this_frame) & (prev_state.respawn_timer == 0)

        reverted_score = jnp.where(collected, prev_state.score, new_state.score)
        reverted_type_idx = jnp.where(
            collected, prev_state.collect_type_index, new_state.collect_type_index
        )
        reverted_type_count = jnp.where(
            collected, prev_state.collect_type_count, new_state.collect_type_count
        )
        cleared_popups = new_state.score_popups.replace(
            x=jnp.full_like(new_state.score_popups.x, -5.0),
            value=jnp.zeros_like(new_state.score_popups.value),
            timer=jnp.zeros_like(new_state.score_popups.timer),
            active=jnp.zeros_like(new_state.score_popups.active),
        )
        score_popups = jax.lax.cond(
            collected, lambda: cleared_popups, lambda: new_state.score_popups
        )

        new_lives = jnp.where(should_die, new_state.lives - 1, new_state.lives)
        new_respawn = jnp.where(
            should_die,
            jnp.int32(self._env.consts.respawn_frames),
            new_state.respawn_timer,
        )
        new_hit = jnp.where(
            should_die,
            jnp.int32(self._env.consts.hit_frames),
            new_state.hit_timer,
        )
        game_over = jnp.where(new_lives <= 0, jnp.array(True), new_state.game_over)

        cleared_enemies = new_state.enemies.replace(
            x=jnp.full_like(new_state.enemies.x, -5.0),
            vx=jnp.zeros_like(new_state.enemies.vx),
            alive=jnp.zeros_like(new_state.enemies.alive),
        )
        cleared_collectibles = new_state.collectibles.replace(
            x=jnp.full_like(new_state.collectibles.x, -5.0),
            vx=jnp.zeros_like(new_state.collectibles.vx),
            alive=jnp.zeros_like(new_state.collectibles.alive),
        )
        enemies = jax.lax.cond(should_die, lambda: cleared_enemies, lambda: new_state.enemies)
        collectibles = jax.lax.cond(
            should_die, lambda: cleared_collectibles, lambda: new_state.collectibles
        )

        return new_state.replace(
            score=reverted_score,
            collect_type_index=reverted_type_idx,
            collect_type_count=reverted_type_count,
            score_popups=score_popups,
            lives=new_lives,
            respawn_timer=new_respawn,
            hit_timer=new_hit,
            game_over=game_over,
            enemies=enemies,
            collectibles=collectibles,
        )


class TopLanesOnlyMod(JaxAtariPostStepModPlugin):
    """Only the top 4 lanes (indices 0–3) may hold entities."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: AsterixState):
        return obs, self._clear_bottom_lanes(state)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: AsterixState, new_state: AsterixState) -> AsterixState:
        del prev_state
        return self._clear_bottom_lanes(new_state)

    @partial(jax.jit, static_argnums=(0,))
    def _clear_bottom_lanes(self, state: AsterixState) -> AsterixState:
        n = state.enemies.alive.shape[0]
        top_mask = jnp.arange(n) < 4
        enemies = state.enemies.replace(
            alive=jnp.where(top_mask, state.enemies.alive, False),
            x=jnp.where(top_mask, state.enemies.x, -5.0),
            vx=jnp.where(top_mask, state.enemies.vx, 0.0),
        )
        collectibles = state.collectibles.replace(
            alive=jnp.where(top_mask, state.collectibles.alive, False),
            x=jnp.where(top_mask, state.collectibles.x, -5.0),
            vx=jnp.where(top_mask, state.collectibles.vx, 0.0),
        )
        # Keep bottom-lane timers from expiring so they never spawn.
        max_delay = jnp.int32(self._env.consts.spawn_max_delay)
        lane_timers = jnp.where(top_mask, state.lane_timers, max_delay)
        return state.replace(
            enemies=enemies,
            collectibles=collectibles,
            lane_timers=lane_timers,
        )


class DenseSpawnsMod(JaxAtariInternalModPlugin):
    """Halve spawn delay range so entities appear more often."""

    constants_overrides = {
        "spawn_min_delay": 10,
        "spawn_max_delay": 25,
    }


class RecolorEnemiesMod(JaxAtariInternalModPlugin):
    """Single-color recolor of enemy (lyre) sprites — visual-only OOD."""

    _ENEMY_COLOR = (0, 220, 220)  # cyan

    asset_overrides = {
        "LYRE_LEFT": {
            "name": "LYRE_LEFT",
            "type": "single",
            "data": _recolor_lyre("LYRE_LEFT.npy", _ENEMY_COLOR),
        },
        "LYRE_RIGHT": {
            "name": "LYRE_RIGHT",
            "type": "single",
            "data": _recolor_lyre("LYRE_RIGHT.npy", _ENEMY_COLOR),
        },
    }
