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


# ---------------------------------------------------------------------------
# Paper-eval grid additions (MODS_PLAN §5 / §7)
# ---------------------------------------------------------------------------

_SCREEN_W = 160

# Remap Atari actions: swap left ↔ right (and diagonals / fire variants).
_LR_SWAP = jnp.array(
    [
        0,   # NOOP
        1,   # FIRE
        2,   # UP
        4,   # RIGHT -> LEFT
        3,   # LEFT -> RIGHT
        5,   # DOWN
        7,   # UPRIGHT -> UPLEFT
        6,   # UPLEFT -> UPRIGHT
        9,   # DOWNRIGHT -> DOWNLEFT
        8,   # DOWNLEFT -> DOWNRIGHT
        10,  # UPFIRE
        12,  # RIGHTFIRE -> LEFTFIRE
        11,  # LEFTFIRE -> RIGHTFIRE
        13,  # DOWNFIRE
        15,  # UPRIGHTFIRE -> UPLEFTFIRE
        14,  # UPLEFTFIRE -> UPRIGHTFIRE
        17,  # DOWNRIGHTFIRE -> DOWNLEFTFIRE
        16,  # DOWNLEFTFIRE -> DOWNRIGHTFIRE
    ],
    dtype=jnp.int32,
)


def _flip_object_obs(obj, screen_w: int = _SCREEN_W):
    """Horizontal flip of an ObjectObservation (left-edge x + facing)."""
    new_x = (screen_w - obj.x - obj.width).astype(obj.x.dtype)
    # Seaquest orientations: 90 = right, 270 = left.
    new_ori = jnp.where(
        obj.orientation == 90.0,
        jnp.float32(270.0),
        jnp.where(obj.orientation == 270.0, jnp.float32(90.0), obj.orientation),
    )
    return obj.replace(x=new_x, orientation=new_ori)


def _action_has_horizontal(action: chex.Array) -> tuple[chex.Array, chex.Array]:
    from jaxatari.environment import JAXAtariAction as Action

    left = jnp.any(
        jnp.array(
            [
                action == Action.LEFT,
                action == Action.UPLEFT,
                action == Action.DOWNLEFT,
                action == Action.LEFTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNLEFTFIRE,
            ]
        )
    )
    right = jnp.any(
        jnp.array(
            [
                action == Action.RIGHT,
                action == Action.UPRIGHT,
                action == Action.DOWNRIGHT,
                action == Action.RIGHTFIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.DOWNRIGHTFIRE,
            ]
        )
    )
    return left, right


class MirrorDiversMod(JaxAtariInternalModPlugin):
    """Decoys: shark slots look like divers and fill empty COLLECT obs slots.

    Spare enemy (shark) slots are rendered with diver sprites, use diver-sized
    observation boxes, and never kill or collect. Empty diver observation slots
    are filled from active sharks so nearest-target COLLECT can latch onto a
    decoy. Real ``diver_positions`` collection logic is unchanged.
    """

    conflicts_with = ["peaceful_enemies", "swap_diver_enemy_labels"]

    asset_overrides = {
        "shark_base": "diver",
    }

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
        # Sharks are decoys (inert); only subs / surface / missiles remain lethal.
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

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SeaquestState):
        from jaxatari.games.jax_seaquest import JaxSeaquest
        from jaxatari.environment import ObjectObservation

        obs = JaxSeaquest._get_observation(self._env, state)
        divers = obs.divers
        enemies = obs.enemies
        c = self._env.consts
        dw = jnp.int32(c.DIVER_SIZE[0])
        dh = jnp.int32(c.DIVER_SIZE[1])

        # Prefer lane-front sharks (indices 0,3,6,9) as decoy sources.
        shark_idx = jnp.array([0, 3, 6, 9], dtype=jnp.int32)
        decoy_x = enemies.x[shark_idx]
        decoy_y = enemies.y[shark_idx]
        decoy_active = enemies.active[shark_idx]
        decoy_ori = enemies.orientation[shark_idx]

        diver_inactive = divers.active == 0
        use_decoy = jnp.logical_and(diver_inactive, decoy_active != 0)

        new_divers = ObjectObservation.create(
            x=jnp.where(use_decoy, decoy_x, divers.x),
            y=jnp.where(use_decoy, decoy_y, divers.y),
            width=jnp.full((4,), dw, dtype=jnp.int32),
            height=jnp.full((4,), dh, dtype=jnp.int32),
            active=jnp.where(use_decoy, decoy_active, divers.active),
            visual_id=divers.visual_id,
            orientation=jnp.where(use_decoy, decoy_ori, divers.orientation),
        )

        # Hide promoted sharks from DESTROY so COLLECT is the only label on decoys.
        promoted = jnp.zeros(enemies.active.shape[0], dtype=bool).at[shark_idx].set(use_decoy)
        new_enemies = enemies.replace(
            active=jnp.where(promoted, jnp.int32(0), enemies.active)
        )
        return obs.replace(divers=new_divers, enemies=new_enemies)


class MirrorWorldMod(JaxAtariInternalModPlugin):
    """Horizontal mirror: flipped obs + L/R action remap (render flipped in controller).

    State stays canonical; agents see a mirrored world. Ego body-relative features
    should be invariant; screen-space policies should collapse.
    """

    conflicts_with = ["inverted_controls"]

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: SeaquestState, action: chex.Array):
        from jaxatari.games.jax_seaquest import JaxSeaquest

        remapped = _LR_SWAP[action.astype(jnp.int32)]
        return JaxSeaquest.player_step(self._env, state, remapped)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SeaquestState):
        from jaxatari.games.jax_seaquest import JaxSeaquest

        obs = JaxSeaquest._get_observation(self._env, state)
        return obs.replace(
            player=_flip_object_obs(obs.player),
            divers=_flip_object_obs(obs.divers),
            enemies=_flip_object_obs(obs.enemies),
            projectiles=_flip_object_obs(obs.projectiles),
        )


class InvertedControlsMod(JaxAtariInternalModPlugin):
    """Permute the action table: left ↔ right (mapper / embodiment probe)."""

    conflicts_with = ["mirror_world", "momentum"]

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: SeaquestState, action: chex.Array):
        from jaxatari.games.jax_seaquest import JaxSeaquest

        remapped = _LR_SWAP[action.astype(jnp.int32)]
        return JaxSeaquest.player_step(self._env, state, remapped)


class MomentumMod(JaxAtariInternalModPlugin):
    """Player inertia: coast horizontally in facing direction when no L/R input."""

    conflicts_with = ["inverted_controls", "mirror_world"]

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: SeaquestState, action: chex.Array):
        from jaxatari.games.jax_seaquest import JaxSeaquest

        player_x, player_y, player_direction = JaxSeaquest.player_step(
            self._env, state, action
        )
        left, right = _action_has_horizontal(action)
        has_lr = jnp.logical_or(left, right)

        # Coast when idle horizontally; facing (player_direction) is ±1 once set.
        facing = jnp.where(state.player_direction == 0, jnp.int32(1), state.player_direction)
        coast_x = state.player_x + facing
        player_x = jnp.where(has_lr, player_x, coast_x)

        bounds = self._env.consts.PLAYER_BOUNDS
        player_x = jnp.clip(player_x, bounds[0, 0], bounds[0, 1])
        return player_x, player_y, player_direction


class SparseWorldMod(JaxAtariPostStepModPlugin):
    """Keep active targets outside the ego 50px distance clip (S4 probe).

    Pushes sharks, subs, and divers that are within 50px of the player outward
    along the player→target vector to just beyond the clip. Density is unchanged.
    """

    MIN_DIST = 51

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        px = new_state.player_x.astype(jnp.float32)
        py = new_state.player_y.astype(jnp.float32)
        min_dist = jnp.float32(self.MIN_DIST)
        y_min = jnp.float32(46)
        y_max = jnp.float32(self._env.consts.PLAYER_BOUNDS[1, 1])

        def _push(pos):
            active = pos[2] != 0
            dx = pos[0].astype(jnp.float32) - px
            dy = pos[1].astype(jnp.float32) - py
            dist = jnp.sqrt(dx * dx + dy * dy)
            # If coincident, push along travel direction (or +x).
            fallback_dx = jnp.where(pos[2] == 0, jnp.float32(1.0), pos[2].astype(jnp.float32))
            dx = jnp.where(dist < 1e-3, fallback_dx, dx)
            dy = jnp.where(dist < 1e-3, jnp.float32(0.0), dy)
            dist = jnp.sqrt(dx * dx + dy * dy)
            scale = min_dist / jnp.maximum(dist, jnp.float32(1e-3))
            too_close = jnp.logical_and(active, dist < min_dist)
            new_x = jnp.where(too_close, (px + dx * scale).astype(pos.dtype), pos[0])
            new_y = jnp.where(
                too_close,
                jnp.clip((py + dy * scale), y_min, y_max).astype(pos.dtype),
                pos[1],
            )
            new_x = jnp.clip(new_x, jnp.int32(-8), jnp.int32(170)).astype(pos.dtype)
            return jnp.where(active, pos.at[0].set(new_x).at[1].set(new_y), pos)

        return new_state.replace(
            shark_positions=jax.vmap(_push)(new_state.shark_positions),
            sub_positions=jax.vmap(_push)(new_state.sub_positions),
            diver_positions=jax.vmap(_push)(new_state.diver_positions),
        )


class TinyDiversMod(JaxAtariInternalModPlugin):
    """Shrink diver hitboxes (S4 sub-φ8 aim probe). Sprites unchanged."""

    constants_overrides = {
        "DIVER_SIZE": jnp.array([4, 5], dtype=jnp.int32),
    }


# ---------------------------------------------------------------------------
# Extra layout / kinematics / object-set probes
# ---------------------------------------------------------------------------


class DynamicLaneDriftMod(JaxAtariPostStepModPlugin):
    """Bucket B: lanes slowly oscillate vertically with per-lane phase.

    Unlike ``vertical_oscillation`` (fast shared wobble), each lane drifts on a
    slow sinusoid so absolute "safe bands" keep moving while relative geometry
    stays coherent for ego.
    """

    conflicts_with = ["vertical_oscillation", "continuous_random_spawns"]

    AMPLITUDE = 10.0
    FREQ = 0.035  # ~180 frames per cycle
    PHASES = (0.0, 1.7, 3.4, 5.1)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        t = new_state.step_counter.astype(jnp.float32)
        phases = jnp.array(self.PHASES, dtype=jnp.float32)
        lane_offsets = (
            self.AMPLITUDE * jnp.sin(t * self.FREQ + phases)
        ).astype(jnp.int32)  # (4,)
        y_min = jnp.int32(46)
        y_max = jnp.int32(self._env.consts.PLAYER_BOUNDS[1, 1])

        def _drift_lane_entities(positions):
            # positions: (12, 3) = 4 lanes × 3 slots
            reshaped = positions.reshape(4, 3, 3)

            def _lane(lane_i, lane_pos):
                off = lane_offsets[lane_i]

                def _one(pos):
                    active = pos[2] != 0
                    new_y = jnp.clip(pos[1] + off, y_min, y_max)
                    return jnp.where(active, pos.at[1].set(new_y), pos)

                return jax.vmap(_one)(lane_pos)

            drifted = jax.vmap(_lane)(jnp.arange(4), reshaped)
            return drifted.reshape(positions.shape)

        def _drift_divers(positions):
            # divers: (4, 3) — one slot per lane
            def _one(lane_i, pos):
                active = pos[2] != 0
                new_y = jnp.clip(pos[1] + lane_offsets[lane_i], y_min, y_max)
                return jnp.where(active, pos.at[1].set(new_y), pos)

            return jax.vmap(_one)(jnp.arange(4), positions)

        return new_state.replace(
            shark_positions=_drift_lane_entities(new_state.shark_positions),
            sub_positions=_drift_lane_entities(new_state.sub_positions),
            diver_positions=_drift_divers(new_state.diver_positions),
        )


class ContinuousRandomSpawnsMod(JaxAtariPostStepModPlugin):
    """Bucket B: enemies keep a continuous random Y instead of the 4-lane grid.

    On spawn (slot inactive→active), sample Y in the playable band. While the
    slot stays active, that Y is held across frames (base movement would other-
    wise snap back to ``SPAWN_POSITIONS_Y``).
    """

    conflicts_with = ["dynamic_lane_drift", "shift_lanes", "vertical_oscillation"]

    Y_MIN = 50
    Y_MAX = 140

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        y_min = jnp.int32(self.Y_MIN)
        y_max = jnp.int32(self.Y_MAX)
        rng = new_state.rng_key

        def _reassign(prev_pos, new_pos, slot_rng):
            was_inactive = prev_pos[2] == 0
            now_active = new_pos[2] != 0
            just_spawned = jnp.logical_and(was_inactive, now_active)
            still_active = jnp.logical_and(jnp.logical_not(was_inactive), now_active)
            rand_y = jax.random.randint(slot_rng, (), y_min, y_max + 1)
            # Hold previous continuous Y while alive; sample on spawn.
            held_y = jnp.where(still_active, prev_pos[1], rand_y)
            new_y = jnp.where(now_active, held_y, new_pos[1])
            # just_spawned uses rand_y; still_active uses prev; inactive keeps 0
            new_y = jnp.where(just_spawned, rand_y, new_y)
            return jnp.where(now_active, new_pos.at[1].set(new_y.astype(new_pos.dtype)), new_pos)

        n_shark = new_state.shark_positions.shape[0]
        n_sub = new_state.sub_positions.shape[0]
        keys = jax.random.split(rng, n_shark + n_sub + 1)
        shark_keys, sub_keys, new_rng = keys[:n_shark], keys[n_shark:n_shark + n_sub], keys[-1]

        new_sharks = jax.vmap(_reassign)(
            prev_state.shark_positions, new_state.shark_positions, shark_keys
        )
        new_subs = jax.vmap(_reassign)(
            prev_state.sub_positions, new_state.sub_positions, sub_keys
        )
        return new_state.replace(
            shark_positions=new_sharks,
            sub_positions=new_subs,
            rng_key=new_rng,
        )


class OceanCurrentsMod(JaxAtariPostStepModPlugin):
    """Bucket C: constant leftward environmental drag on the player."""

    conflicts_with = ["gravity"]

    DRIFT_PERIOD = 3  # one pixel left every N frames

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        tick = new_state.step_counter % self.DRIFT_PERIOD == 0
        # Only apply underwater so surface deposit / oxygen logic stays sane.
        underwater = new_state.player_y > jnp.int32(52)
        should_drift = jnp.logical_and(tick, underwater)
        bounds = self._env.consts.PLAYER_BOUNDS
        new_x = jnp.where(
            should_drift,
            jnp.maximum(new_state.player_x - 1, bounds[0, 0]),
            new_state.player_x,
        )
        return new_state.replace(player_x=new_x)


class ErraticEnemiesMod(JaxAtariPostStepModPlugin):
    """Bucket C: per-enemy pause / burst — breaks frame-stack timing.

    Each active shark/sub periodically pauses (undo base motion) or bursts
    (+2px extra). Deterministic from (step, slot) so JIT stays pure.
    """

    conflicts_with = ["faster_enemies", "slower_enemies"]

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        step = new_state.step_counter

        def _erratic(pos, slot):
            active = pos[2] != 0
            # 16-beat cycle, phase-shifted per slot.
            beat = (step + slot * 7) % 16
            pause = beat < 4
            burst = beat >= 12
            # Pause: pull back one step of facing motion; burst: push +2.
            delta = jnp.where(pause, -pos[2], jnp.where(burst, pos[2] * 2, 0))
            new_x = pos[0] + delta
            return jnp.where(active, pos.at[0].set(new_x), pos)

        shark_slots = jnp.arange(new_state.shark_positions.shape[0])
        sub_slots = jnp.arange(new_state.sub_positions.shape[0]) + 12
        return new_state.replace(
            shark_positions=jax.vmap(_erratic)(new_state.shark_positions, shark_slots),
            sub_positions=jax.vmap(_erratic)(new_state.sub_positions, sub_slots),
        )


class MicroSwarmMod(JaxAtariPostStepModPlugin):
    """Bucket D: pack all enemy slots with tiny hitboxes (S3 nearest-target stress).

    Shrinks shark/sub collision boxes and fills empty slots by cloning the lane
    front entity with staggered X offsets — many equally close micro-threats.
    """

    conflicts_with = ["only_sharks", "only_submarines", "disable_enemies", "no_enemies"]

    constants_overrides = {
        "SHARK_SIZE": jnp.array([3, 3], dtype=jnp.int32),
        "ENEMY_SUB_SIZE": jnp.array([3, 3], dtype=jnp.int32),
    }

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: SeaquestState):
        timers = jnp.maximum(state.spawn_state.spawn_timers // 3, jnp.int32(40))
        spawn = state.spawn_state.replace(spawn_timers=timers)
        return obs, state.replace(spawn_state=spawn)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        timers = jnp.maximum(new_state.spawn_state.spawn_timers // 3, jnp.int32(40))
        spawn = new_state.spawn_state.replace(spawn_timers=timers)

        def _swarm_fill(positions):
            # (12, 3) → (4, 3, 3)
            lanes = positions.reshape(4, 3, 3)

            def _fill_lane(lane_pos):
                active = lane_pos[:, 2] != 0
                any_active = jnp.any(active)
                # Lane front: first active slot (prefer index 0 then 1 then 2).
                front = jnp.where(
                    active[0],
                    lane_pos[0],
                    jnp.where(active[1], lane_pos[1], lane_pos[2]),
                )

                def _slot(i, pos):
                    # Stagger clones ±12px around the front entity.
                    offset = jnp.array([-12, 0, 12], dtype=pos.dtype)[i]
                    clone = front.at[0].set(front[0] + offset)
                    should_fill = jnp.logical_and(any_active, pos[2] == 0)
                    return jnp.where(should_fill, clone, pos)

                return jax.vmap(_slot)(jnp.arange(3), lane_pos)

            return jax.vmap(_fill_lane)(lanes).reshape(positions.shape)

        return new_state.replace(
            spawn_state=spawn,
            shark_positions=_swarm_fill(new_state.shark_positions),
            sub_positions=_swarm_fill(new_state.sub_positions),
        )


# ---------------------------------------------------------------------------
# High-leverage extras (layout / kinematics / object-set / embodiment)
# ---------------------------------------------------------------------------


class StickyWaterMod(JaxAtariInternalModPlugin):
    """Bucket C: half player speed underwater (embodiment drag, no current)."""

    conflicts_with = ["momentum", "ocean_currents"]

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: SeaquestState, action: chex.Array):
        from jaxatari.games.jax_seaquest import JaxSeaquest

        player_x, player_y, player_direction = JaxSeaquest.player_step(
            self._env, state, action
        )
        underwater = state.player_y > jnp.int32(52)
        # Move only on even frames while submerged → ~half speed.
        apply_move = jnp.logical_or(
            jnp.logical_not(underwater),
            state.step_counter % 2 == 0,
        )
        player_x = jnp.where(apply_move, player_x, state.player_x)
        player_y = jnp.where(apply_move, player_y, state.player_y)
        return player_x, player_y, player_direction


class GhostSharksMod(JaxAtariInternalModPlugin):
    """Bucket D: sharks stay visible DESTROY targets but never kill or die.

    False positives for nearest-DESTROY: contact is harmless and player
    missiles pass through (no reward change — scoring path untouched).
    """

    conflicts_with = ["peaceful_enemies", "mirror_divers", "only_submarines"]

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

    @partial(jax.jit, static_argnums=(0,))
    def check_missile_collisions(
        self,
        missile_pos,
        shark_positions,
        sub_positions,
        score,
        successful_rescues,
        spawn_state,
        rng_key,
    ):
        """Player torpedoes ignore sharks; subs still die normally."""
        from jaxatari.games.jax_seaquest import JaxSeaquest

        (
            new_missile,
            _ignored_sharks,
            new_subs,
            new_score,
            new_spawn_state,
            new_rng,
        ) = JaxSeaquest.check_missile_collisions(
            self._env,
            missile_pos,
            jnp.zeros_like(shark_positions),
            sub_positions,
            score,
            successful_rescues,
            spawn_state,
            rng_key,
        )
        return (
            new_missile,
            shark_positions,
            new_subs,
            new_score,
            new_spawn_state,
            new_rng,
        )


class SlipperyTurnMod(JaxAtariInternalModPlugin):
    """Bucket F: facing updates only every N frames (strafe ≠ turn preview).

    Locomotion still follows the action; ``player_direction`` lags so the
    facing-frame mapper and pixel heading disagree more often.
    """

    conflicts_with = ["momentum", "inverted_controls", "mirror_world", "sticky_water"]

    TURN_PERIOD = 8

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: SeaquestState, action: chex.Array):
        from jaxatari.games.jax_seaquest import JaxSeaquest

        player_x, player_y, commanded_dir = JaxSeaquest.player_step(
            self._env, state, action
        )
        can_turn = jnp.logical_or(
            state.step_counter % self.TURN_PERIOD == 0,
            state.player_direction == 0,
        )
        player_direction = jnp.where(can_turn, commanded_dir, state.player_direction)
        return player_x, player_y, player_direction


class LaneScrambleMod(JaxAtariPostStepModPlugin):
    """Bucket B: periodically permute which lane owns which Y band.

    Discrete reshuffle (not slow drift): absolute safe-zones jump; relative
    within-lane structure stays intact.
    """

    conflicts_with = [
        "dynamic_lane_drift",
        "continuous_random_spawns",
        "shift_lanes",
        "vertical_oscillation",
    ]

    PERIOD = 90
    # Fixed set of permutations of lanes [0,1,2,3].
    PERMS = (
        (0, 1, 2, 3),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 1, 0),
        (1, 2, 3, 0),
        (3, 0, 1, 2),
    )

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        del prev_state
        base_ys = self._env.consts.SPAWN_POSITIONS_Y.astype(jnp.int32)
        perms = jnp.array(self.PERMS, dtype=jnp.int32)
        epoch = (new_state.step_counter // self.PERIOD) % perms.shape[0]
        perm = perms[epoch]  # lane i → band perm[i]

        def _scramble_enemies(positions):
            lanes = positions.reshape(4, 3, 3)

            def _lane(lane_i, lane_pos):
                src_y = base_ys[lane_i]
                dst_y = base_ys[perm[lane_i]]
                delta = dst_y - src_y

                def _one(pos):
                    active = pos[2] != 0
                    return jnp.where(active, pos.at[1].set(pos[1] + delta), pos)

                return jax.vmap(_one)(lane_pos)

            return jax.vmap(_lane)(jnp.arange(4), lanes).reshape(positions.shape)

        def _scramble_divers(positions):
            def _one(lane_i, pos):
                active = pos[2] != 0
                delta = base_ys[perm[lane_i]] - base_ys[lane_i]
                return jnp.where(active, pos.at[1].set(pos[1] + delta), pos)

            return jax.vmap(_one)(jnp.arange(4), positions)

        return new_state.replace(
            shark_positions=_scramble_enemies(new_state.shark_positions),
            sub_positions=_scramble_enemies(new_state.sub_positions),
            diver_positions=_scramble_divers(new_state.diver_positions),
        )


