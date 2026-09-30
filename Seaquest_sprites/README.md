# Seaquest ALE-synced assets (robustness tree)

## Collected diver indicator
- **Runtime path (what the renderer loads):**
  `~/.local/share/jaxatari/sprites/seaquest/diver_indicator/1.npy`
- **Repo copy:** `Seaquest_sprites/diver_indicator/1.npy` (9×8 RGBA, ALE OC)
- Extracted from `seaquest_play01.npz` CollectedDiver at (58, 178).
- Backup of the old 7×8 glyph: `1.npy.bak_pre_ale_sync` next to the runtime file.

If you reinstall sprites from another package and lose the ALE indicator, copy from `Seaquest_sprites/diver_indicator/1.npy` into the runtime path above.

## Surface-wave bake inputs
- `Seaquest_screenshots/frame_*.npy` — one frame per ALE wave hold (t=0,1,9,17,…) from `seaquest_play01.npz`, used by `SeaquestRenderer._bake_surface_wave_backgrounds`.
