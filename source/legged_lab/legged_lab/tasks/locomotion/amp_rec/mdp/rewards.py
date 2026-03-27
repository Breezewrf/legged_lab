from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

from legged_lab.tasks.locomotion.amp.mdp.rewards import *
from .fallen_state import fallen_fast_mask, torso_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp_when_upright(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    fallen_min_duration_s: float = 0.25,
    fallen_height_threshold: float = 0.26,
    fallen_velocity_scale: float = 0.3,
    fall_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track velocity only when not in fallen recovery mode."""
    reward = mdp.track_lin_vel_xy_exp(env, command_name=command_name, std=std, asset_cfg=asset_cfg)
    fallen = fallen_fast_mask(
        env,
        min_duration_s=fallen_min_duration_s,
        height_threshold=fallen_height_threshold,
        asset_cfg=fall_asset_cfg,
    )
    reward_scale = torch.where(
        fallen,
        torch.full_like(reward, fallen_velocity_scale),
        torch.ones_like(reward),
    )
    return reward * reward_scale


def upright_recovery_reward(
    env: ManagerBasedRLEnv,
    fallen_min_duration_s: float = 0.25,
    fallen_height_threshold: float = 0.26,
    up_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Reward torso alignment with world up while recovering from a fall."""
    asset: RigidObject = env.scene[up_asset_cfg.name]

    body_quat_w = asset.data.body_quat_w[:, up_asset_cfg.body_ids[0], :]
    body_up = math_utils.quat_apply(
        body_quat_w,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, -1),
    )
    upright_score = torch.clamp(body_up[:, 2], min=0.0, max=1.0)

    fallen = fallen_fast_mask(
        env,
        min_duration_s=fallen_min_duration_s,
        height_threshold=fallen_height_threshold,
        asset_cfg=up_asset_cfg,
    )
    return upright_score * fallen.float()


def root_height_recovery_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    fallen_min_duration_s: float = 0.25,
    fallen_height_threshold: float = 0.26,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Reward torso height increase toward target while recovering from fall."""
    height = torso_height(env, asset_cfg)
    height_error = target_height - height
    height_score = torch.exp(-torch.square(height_error) / (std**2))

    fallen = fallen_fast_mask(
        env,
        min_duration_s=fallen_min_duration_s,
        height_threshold=fallen_height_threshold,
        asset_cfg=asset_cfg,
    )
    return height_score * fallen.float()
