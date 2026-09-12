import jax
import jax.numpy as jnp
import chex
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_seaquest import SeaquestState, SpawnState


class DisableEnemiesMod(JaxAtariPostStepModPlugin):
    """Disable enemies in the environment."""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        # Zero out all enemy positions
        return new_state.replace(
            shark_positions=jnp.zeros_like(new_state.shark_positions),
            sub_positions=jnp.zeros_like(new_state.sub_positions),
            enemy_missile_positions=jnp.zeros_like(new_state.enemy_missile_positions),
            surface_sub_position=jnp.zeros_like(new_state.surface_sub_position)
        )


class NoDiversMod(JaxAtariInternalModPlugin):
    """
    Internal mod to remove Divers from the game.
    It suppresses the logic that updates/spawns divers and disables their rendering.
    """

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step_diver_movement(self,
            diver_positions: chex.Array,
            shark_positions: chex.Array,
            state_player_x: chex.Array,
            state_player_y: chex.Array,
            state_divers_collected: chex.Array,
            spawn_state: SpawnState,
            step_counter: chex.Array,
            rng: chex.PRNGKey
        ):
        """
        Override for step_diver_movement: clear all diver slots.

        Diver activity is ``direction != 0`` (see jax_seaquest._get_observation).
        Filling with ``-1`` left divers *active* at clipped (0, 0) — a ceiling
        COLLECT ghost. Use zeros so slots are inactive, matching DisableEnemiesMod.
        """
        return (
            jnp.zeros_like(diver_positions),
            state_divers_collected,
            spawn_state,
            rng,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _draw_divers(self, raster: jnp.ndarray, state: SeaquestState):
        """
        Override for the renderer to skip drawing divers.
        """
        # Simply return the raster without drawing the sprite
        return raster


class EnemyMinesMod(JaxAtariInternalModPlugin):
    """
    Replaces both Sharks and Enemy Submarines with Mine sprites.
    
    This is a visual-only mod. Hitboxes and movement logic remain identical 
    to the original enemies. The 'Sharks' (now Mines) will not change color 
    based on difficulty level due to the game's rendering logic.
    """

    asset_overrides = {
        "shark_base": {
            'name': 'shark_base',
            'type': 'group',
            'files': ['mods/mine.npy', 'mods/mine.npy']
        },
        "enemy_sub": {
            'name': 'enemy_sub',
            'type': 'group',
            'files': ['mods/mine.npy', 'mods/mine.npy']
        }
    }

    constants_overrides = {
        "SHARK_DIFFICULTY_COLORS": jnp.array([[128, 128, 128]] * 5),
    }


class FireBallsMod(JaxAtariInternalModPlugin):
    """
    Replaces both Sharks and Enemy Submarines with Mine sprites.
    
    This is a visual-only mod. Hitboxes and movement logic remain identical 
    to the original enemies. The 'Sharks' (now Mines) will not change color 
    based on difficulty level due to the game's rendering logic.
    """

class UnlimitedOxygenMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        return new_state.replace(oxygen=jnp.array(64, dtype=jnp.int32))

class GravityMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        new_player_y = jnp.where(
            new_state.step_counter % 4 == 0,
            jnp.minimum(new_state.player_y + 1, self._env.consts.PLAYER_BOUNDS[1, 1]),
            new_state.player_y
        )
        return new_state.replace(player_y=new_player_y)

class RandomColorEnemiesMod(JaxAtariInternalModPlugin):
    pass


def _boost_enemy_x(positions: chex.Array, extra: int) -> chex.Array:
    """Apply extra horizontal movement to active enemy slots (shape N×3)."""

    def _boost_one(pos):
        active = pos[2] != 0
        new_x = pos[0] + pos[2] * extra
        return jnp.where(active, pos.at[0].set(new_x), pos)

    return jax.vmap(_boost_one)(positions)


def _oscillate_enemy_y(positions: chex.Array, offset: chex.Array, y_min: int, y_max: int) -> chex.Array:
    def _osc_one(pos):
        active = pos[2] != 0
        new_y = jnp.clip(pos[1] + offset, y_min, y_max)
        return jnp.where(active, pos.at[1].set(new_y), pos)

    return jax.vmap(_osc_one)(positions)


class PeacefulEnemiesMod(JaxAtariInternalModPlugin):
    """Enemies remain but player collisions no longer cause death or bonus points."""

    @partial(jax.jit, static_argnums=(0,))
    def check_player_collision(
        self,
        player_x,
        player_y,
        submarine_list,
        shark_list,
        surface_sub_pos,
        enemy_projectile_list,
        score,
        successful_rescues,
    ):
        del player_x, player_y, submarine_list, shark_list, surface_sub_pos
        del enemy_projectile_list, score, successful_rescues
        return jnp.array(False, dtype=jnp.bool_), jnp.array(0, dtype=jnp.int32)


class NoEnemyTorpedoesMod(JaxAtariPostStepModPlugin):
    """Remove enemy torpedoes after each step (collision-only threats remain)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        return new_state.replace(
            enemy_missile_positions=jnp.zeros_like(new_state.enemy_missile_positions)
        )


class FasterEnemiesMod(JaxAtariPostStepModPlugin):
    """Enemies move one extra pixel per step in their travel direction."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        return new_state.replace(
            shark_positions=_boost_enemy_x(new_state.shark_positions, 1),
            sub_positions=_boost_enemy_x(new_state.sub_positions, 1),
        )


class SlowerEnemiesMod(JaxAtariPostStepModPlugin):
    """Enemies move one fewer pixel on alternating frames (roughly half speed)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        undo = jnp.where(new_state.step_counter % 2 == 0, 1, 0)
        return new_state.replace(
            shark_positions=_boost_enemy_x(new_state.shark_positions, -undo),
            sub_positions=_boost_enemy_x(new_state.sub_positions, -undo),
        )


class ShiftLanesMod(JaxAtariInternalModPlugin):
    """Shift enemy lane Y positions downward by 10 pixels."""

    constants_overrides = {
        "SPAWN_POSITIONS_Y": jnp.array([81, 105, 129, 149], dtype=jnp.int32),
        "ENEMY_MISSILE_Y": jnp.array([83, 107, 131, 151], dtype=jnp.int32),
    }


class OnlySubmarinesMod(JaxAtariPostStepModPlugin):
    """Force all lanes to spawn enemy submarines instead of sharks."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: SeaquestState):
        spawn = state.spawn_state.replace(prev_sub=jnp.ones(4, dtype=jnp.int32))
        return obs, state.replace(spawn_state=spawn)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        spawn = new_state.spawn_state.replace(prev_sub=jnp.ones(4, dtype=jnp.int32))
        zero_sharks = jnp.zeros_like(new_state.shark_positions)
        return new_state.replace(shark_positions=zero_sharks, spawn_state=spawn)


class OnlySharksMod(JaxAtariPostStepModPlugin):
    """Force all lanes to spawn sharks instead of enemy submarines."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: SeaquestState):
        spawn = state.spawn_state.replace(prev_sub=jnp.zeros(4, dtype=jnp.int32))
        return obs, state.replace(spawn_state=spawn)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        spawn = new_state.spawn_state.replace(prev_sub=jnp.zeros(4, dtype=jnp.int32))
        zero_subs = jnp.zeros_like(new_state.sub_positions)
        return new_state.replace(sub_positions=zero_subs, spawn_state=spawn)


class VerticalOscillationMod(JaxAtariPostStepModPlugin):
    """Sharks, subs, and divers oscillate vertically (±3 px)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        offset = (3.0 * jnp.sin(new_state.step_counter.astype(jnp.float32) * 0.15)).astype(
            jnp.int32
        )
        y_min = jnp.int32(46)
        y_max = jnp.int32(self._env.consts.PLAYER_BOUNDS[1, 1])
        return new_state.replace(
            shark_positions=_oscillate_enemy_y(new_state.shark_positions, offset, y_min, y_max),
            sub_positions=_oscillate_enemy_y(new_state.sub_positions, offset, y_min, y_max),
            diver_positions=_oscillate_enemy_y(new_state.diver_positions, offset, y_min, y_max),
        )


class DenseSpawnsMod(JaxAtariPostStepModPlugin):
    """Halve spawn timers so enemies appear more frequently."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: SeaquestState):
        timers = jnp.maximum(state.spawn_state.spawn_timers // 2, jnp.int32(80))
        spawn = state.spawn_state.replace(spawn_timers=timers)
        return obs, state.replace(spawn_state=spawn)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        timers = jnp.maximum(new_state.spawn_state.spawn_timers // 2, jnp.int32(80))
        spawn = new_state.spawn_state.replace(spawn_timers=timers)
        return new_state.replace(spawn_state=spawn)


class FastOxygenDrainMod(JaxAtariPostStepModPlugin):
    """Additional oxygen drain underwater every 16 frames (on top of base drain)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        underwater = new_state.player_y > jnp.int32(52)
        extra_tick = jnp.logical_and(
            underwater,
            jnp.logical_and(
                new_state.step_counter % 16 == 0,
                new_state.step_counter % 32 != 0,
            ),
        )
        new_oxygen = jnp.where(
            extra_tick,
            jnp.maximum(new_state.oxygen - 1, jnp.int32(0)),
            new_state.oxygen,
        )
        return new_state.replace(oxygen=new_oxygen)


class SlowOxygenDrainMod(JaxAtariPostStepModPlugin):
    """Refill one oxygen unit on half of the normal underwater drain ticks."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        underwater = new_state.player_y > jnp.int32(52)
        would_drain = jnp.logical_and(underwater, new_state.step_counter % 32 == 0)
        skip_drain = jnp.logical_and(would_drain, new_state.step_counter % 64 != 0)
        new_oxygen = jnp.where(
            skip_drain,
            jnp.minimum(new_state.oxygen + 1, jnp.int32(64)),
            new_state.oxygen,
        )
        return new_state.replace(oxygen=new_oxygen)


class SurfaceSubAlwaysMod(JaxAtariPostStepModPlugin):
    """Keep the surface submarine active near the top of the screen."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        # Match env dtype (reset uses jnp.zeros → float32); int32 here breaks noop-reset cond.
        dtype = new_state.surface_sub_position.dtype
        active_sub = jnp.array([159, 45, -1], dtype=dtype)
        return new_state.replace(surface_sub_position=active_sub)


class InvertedLanesMod(JaxAtariPostStepModPlugin):
    """Flip default lane travel directions."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: SeaquestState):
        flipped = 1 - state.spawn_state.lane_directions
        spawn = state.spawn_state.replace(lane_directions=flipped.astype(jnp.int32))
        return obs, state.replace(spawn_state=spawn)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        flipped = 1 - new_state.spawn_state.lane_directions
        spawn = new_state.spawn_state.replace(lane_directions=flipped.astype(jnp.int32))
        return new_state.replace(spawn_state=spawn)


class LethalDiversMod(JaxAtariPostStepModPlugin):
    """Diver contact triggers the death animation (game mod only; CBL labels still COLLECT)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state

        def _diver_hits(diver_pos):
            active = diver_pos[2] != 0
            hit = self._env.check_collision_single(
                jnp.array([new_state.player_x, new_state.player_y]),
                self._env.consts.PLAYER_SIZE,
                diver_pos,
                self._env.consts.DIVER_SIZE,
            )
            return jnp.logical_and(active, hit)

        diver_hit = jnp.any(jax.vmap(_diver_hits)(new_state.diver_positions))
        already_dying = new_state.death_counter > 0
        should_die = jnp.logical_and(diver_hit, jnp.logical_not(already_dying))
        new_spawn = jax.lax.cond(
            should_die,
            lambda s: self._env.soft_reset_spawn_state(s),
            lambda s: s,
            new_state.spawn_state,
        )
        return new_state.replace(
            death_counter=jnp.where(should_die, jnp.int32(90), new_state.death_counter),
            spawn_state=new_spawn,
        )


# Backwards-compatible alias used in some experiment notes.
NoEnemiesMod = DisableEnemiesMod


class PenalizeDiverShootingMod(JaxAtariPostStepModPlugin):
    """
    Penalizes the player for shooting divers with the torpedo.

    When the player's missile hits a diver:
    - The diver is removed
    - The missile is consumed
    - Score is penalized proportionally to difficulty:
        penalty = min(PENALTY_BASE + PENALTY_STEP * successful_rescues, PENALTY_MAX)
    """

    PENALTY_BASE = 50
    PENALTY_STEP = 25
    PENALTY_MAX = 500

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        missile_pos = new_state.player_missile_position
        missile_active = missile_pos[2] != 0

        penalty = jnp.minimum(
            self.PENALTY_BASE + self.PENALTY_STEP * new_state.successful_rescues,
            self.PENALTY_MAX,
        )

        missile_xy = missile_pos[:2]
        missile_size = self._env.consts.MISSILE_SIZE
        diver_size = self._env.consts.DIVER_SIZE

        def check_diver(i, carry):
            state, missile_gone = carry
            diver_pos = state.diver_positions[i]

            should_check = jnp.logical_and(diver_pos[2] != 0, jnp.logical_not(missile_gone))
            collision = self._env.check_collision_single(
                missile_xy,
                missile_size,
                jnp.array([diver_pos[0], diver_pos[1]]),
                diver_size,
            )
            hit = jnp.logical_and(should_check, collision)

            state = state.replace(
                diver_positions=state.diver_positions.at[i].set(
                    jnp.where(hit, jnp.zeros(3, dtype=diver_pos.dtype), diver_pos)
                ),
                score=jnp.where(hit, state.score - penalty, state.score),
                player_missile_position=jnp.where(
                    hit,
                    jnp.zeros(3, dtype=state.player_missile_position.dtype),
                    state.player_missile_position,
                ),
            )
            return state, jnp.logical_or(missile_gone, hit)

        return jax.lax.cond(
            missile_active,
            lambda s: jax.lax.fori_loop(0, 4, check_diver, (s, jnp.array(False)))[0],
            lambda s: s,
            new_state,
        )


class PeacefulSharksOnlyMod(JaxAtariInternalModPlugin):
    """Internal helper: shark contact no longer kills (subs/missiles/surface still do)."""

    conflicts_with = ["peaceful_enemies"]

    @partial(jax.jit, static_argnums=(0,))
    def check_player_collision(
        self,
        player_x,
        player_y,
        submarine_list,
        shark_list,
        surface_sub_pos,
        enemy_projectile_list,
        score,
        successful_rescues,
    ):
        del shark_list, score
        submarine_collisions = jnp.any(
            self._env.check_collision_batch(
                jnp.array([player_x, player_y]),
                self._env.consts.PLAYER_SIZE,
                submarine_list,
                self._env.consts.ENEMY_SUB_SIZE,
            )
        )
        surface_collision = self._env.check_collision_single(
            jnp.array([player_x, player_y]),
            self._env.consts.PLAYER_SIZE,
            surface_sub_pos,
            self._env.consts.ENEMY_SUB_SIZE,
        )
        missile_collisions = jnp.any(
            self._env.check_collision_batch(
                jnp.array([player_x, player_y]),
                self._env.consts.PLAYER_SIZE,
                enemy_projectile_list,
                self._env.consts.MISSILE_SIZE,
            )
        )
        collision_points = jnp.where(
            submarine_collisions,
            self._env.calculate_kill_points(successful_rescues),
            jnp.where(
                surface_collision,
                self._env.calculate_kill_points(successful_rescues),
                0,
            ),
        )
        died = jnp.any(
            jnp.array([submarine_collisions, missile_collisions, surface_collision])
        )
        return died, collision_points


class CollectSharksOnContactMod(JaxAtariPostStepModPlugin):
    """Post-step helper: touching a shark fills a diver slot (reverse affordance)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        already_dying = new_state.death_counter > 0

        def _collect_one(i, state):
            shark = state.shark_positions[i]
            active = shark[2] != 0
            can_collect = state.divers_collected < 6
            hit = self._env.check_collision_single(
                jnp.array([state.player_x, state.player_y]),
                self._env.consts.PLAYER_SIZE,
                shark,
                self._env.consts.SHARK_SIZE,
            )
            should = jnp.logical_and(
                jnp.logical_and(jnp.logical_and(active, hit), can_collect),
                jnp.logical_not(already_dying),
            )
            return state.replace(
                shark_positions=state.shark_positions.at[i].set(
                    jnp.where(should, jnp.zeros_like(shark), shark)
                ),
                divers_collected=jnp.where(
                    should, state.divers_collected + 1, state.divers_collected
                ),
            )

        return jax.lax.fori_loop(0, new_state.shark_positions.shape[0], _collect_one, new_state)


class SwapDiverEnemyLabelsMod(JaxAtariInternalModPlugin):
    """
    Full semantic swap of diver ↔ enemy (shark) labels:
    - sprites swapped (divers look like sharks and vice versa)
    - object-centric observation channels swapped (first 4 enemies ↔ divers)
    """

    asset_overrides = {
        "shark_base": "diver",
        "diver": "shark_base",
    }

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SeaquestState):
        from jaxatari.games.jax_seaquest import JaxSeaquest
        from jaxatari.environment import ObjectObservation

        obs = JaxSeaquest._get_observation(self._env, state)
        enemies = obs.enemies
        divers = obs.divers

        new_divers = ObjectObservation.create(
            x=enemies.x[:4],
            y=enemies.y[:4],
            width=enemies.width[:4],
            height=enemies.height[:4],
            active=enemies.active[:4],
            visual_id=enemies.visual_id[:4],
            orientation=enemies.orientation[:4],
        )
        new_enemies = ObjectObservation.create(
            x=jnp.concatenate([divers.x, enemies.x[4:]]),
            y=jnp.concatenate([divers.y, enemies.y[4:]]),
            width=jnp.concatenate([divers.width, enemies.width[4:]]),
            height=jnp.concatenate([divers.height, enemies.height[4:]]),
            active=jnp.concatenate([divers.active, enemies.active[4:]]),
            visual_id=jnp.concatenate(
                [jnp.zeros_like(divers.visual_id), enemies.visual_id[4:]]
            ),
            orientation=jnp.concatenate([divers.orientation, enemies.orientation[4:]]),
        )
        return obs.replace(divers=new_divers, enemies=new_enemies)


class ExtraEnemyTypeMod(JaxAtariPostStepModPlugin):
    """
    Compositional OOD: mine visuals + synchronized multi-lane spawn bursts
    (new enemy appearance and a new spawn pattern together).
    """

    asset_overrides = {
        "shark_base": {
            "name": "shark_base",
            "type": "group",
            "files": ["mods/mine.npy", "mods/mine.npy"],
        },
        "enemy_sub": {
            "name": "enemy_sub",
            "type": "group",
            "files": ["mods/mine.npy", "mods/mine.npy"],
        },
    }

    constants_overrides = {
        "SHARK_DIFFICULTY_COLORS": jnp.array([[128, 128, 128]] * 5),
    }

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: SeaquestState):
        timers = state.spawn_state.spawn_timers
        synced = jnp.full_like(timers, jnp.min(timers))
        spawn = state.spawn_state.replace(spawn_timers=synced)
        return obs, state.replace(spawn_state=spawn)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        timers = new_state.spawn_state.spawn_timers
        synced = jnp.full_like(timers, jnp.min(timers))
        spawn = new_state.spawn_state.replace(spawn_timers=synced)

        # Extra vertical wobble on top of base shark motion (new motion pattern).
        offset = (2.0 * jnp.sin(new_state.step_counter.astype(jnp.float32) * 0.25)).astype(
            jnp.int32
        )
        y_min = jnp.int32(46)
        y_max = jnp.int32(self._env.consts.PLAYER_BOUNDS[1, 1])

        def _wobble(pos):
            active = pos[2] != 0
            new_y = jnp.clip(pos[1] + offset, y_min, y_max)
            return jnp.where(active, pos.at[1].set(new_y), pos)

        return new_state.replace(
            spawn_state=spawn,
            shark_positions=jax.vmap(_wobble)(new_state.shark_positions),
            sub_positions=jax.vmap(_wobble)(new_state.sub_positions),
        )


class NoSurfacingDeathMod(JaxAtariInternalModPlugin):
    """Surfacing with 0 divers no longer costs a life.

    Collisions, oxygen-out, and the 1–5 diver deposit tax are unchanged.
    Divers collected is clamped at 0 so a 0-diver surface does not go negative.
    """

    @partial(jax.jit, static_argnums=(0,))
    def update_oxygen(self, state, player_x, player_y, player_missile_position):
        from jaxatari.games.jax_seaquest import JaxSeaquest

        (
            new_oxygen,
            player_x,
            player_y,
            player_missile_position,
            oxygen_depleted,
            _lose_life,
            new_divers_collected,
            should_reset,
            new_just_surfaced,
            new_difficulty,
        ) = JaxSeaquest.update_oxygen(
            self._env, state, player_x, player_y, player_missile_position
        )
        del _lose_life
        return (
            new_oxygen,
            player_x,
            player_y,
            player_missile_position,
            oxygen_depleted,
            jnp.array(False),
            jnp.maximum(new_divers_collected, jnp.int32(0)),
            should_reset,
            new_just_surfaced,
            new_difficulty,
        )


