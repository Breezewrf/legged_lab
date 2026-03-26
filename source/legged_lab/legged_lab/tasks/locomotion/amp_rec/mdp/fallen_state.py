from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def torso_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Return torso height in world frame for each environment."""
    asset: RigidObject = env.scene[asset_cfg.name]
    heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if heights.ndim == 1:
        return heights
    return heights.mean(dim=1)


def _ensure_fallen_buffers(env: ManagerBasedEnv):
    if hasattr(env, "_amp_rec_fallen_steps"):
        return
    env._amp_rec_fallen_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._amp_rec_last_update = -1


def _update_fallen_steps(
    env: ManagerBasedEnv,
    heights: torch.Tensor,
    height_threshold: float,
):
    _ensure_fallen_buffers(env)

    current_step = int(env.common_step_counter)
    if env._amp_rec_last_update == current_step:
        return

    # Reset per-env counters immediately after environment reset.
    just_reset = env.episode_length_buf == 0
    if torch.any(just_reset):
        env._amp_rec_fallen_steps[just_reset] = 0

    below = heights < height_threshold
    env._amp_rec_fallen_steps = torch.where(
        below,
        env._amp_rec_fallen_steps + 1,
        torch.zeros_like(env._amp_rec_fallen_steps),
    )
    env._amp_rec_last_update = current_step


def fallen_time_mask(
    env: ManagerBasedEnv,
    min_duration_s: float,
    height_threshold: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Return boolean mask when torso stays below threshold for at least min_duration_s."""
    heights = torso_height(env, asset_cfg)
    _update_fallen_steps(env, heights, height_threshold)

    min_steps = max(1, int(math.ceil(min_duration_s / env.step_dt)))
    return env._amp_rec_fallen_steps >= min_steps


def fallen_fast_mask(
    env: ManagerBasedEnv,
    min_duration_s: float = 0.25,
    height_threshold: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Fast fallen detector for reward gating."""
    return fallen_time_mask(
        env=env,
        min_duration_s=min_duration_s,
        height_threshold=height_threshold,
        asset_cfg=asset_cfg,
    )


def fallen_persistent_mask(
    env: ManagerBasedEnv,
    min_duration_s: float = 2.0,
    height_threshold: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Persistent fallen detector for robust state labeling."""
    return fallen_time_mask(
        env=env,
        min_duration_s=min_duration_s,
        height_threshold=height_threshold,
        asset_cfg=asset_cfg,
    )
