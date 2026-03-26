from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

from legged_lab.tasks.locomotion.amp.mdp.observations import *
from .fallen_state import fallen_fast_mask, fallen_persistent_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def contact_force_norm(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Contact force norm (max over history) for selected bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_force_hist = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_norm = torch.linalg.norm(net_force_hist, dim=-1).max(dim=1)[0]
    return force_norm.reshape(env.num_envs, -1)


def contact_binary(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact indicator for selected bodies."""
    force_norm = contact_force_norm(env, sensor_cfg)
    return (force_norm > threshold).float()


def fallen_fast_obs(
    env: ManagerBasedEnv,
    min_duration_s: float = 0.25,
    height_threshold: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Fast fallen detector as observation."""
    return fallen_fast_mask(
        env=env,
        min_duration_s=min_duration_s,
        height_threshold=height_threshold,
        asset_cfg=asset_cfg,
    ).float().unsqueeze(-1)


def fallen_persistent_obs(
    env: ManagerBasedEnv,
    min_duration_s: float = 2.0,
    height_threshold: float = 0.32,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Persistent fallen detector as observation."""
    return fallen_persistent_mask(
        env=env,
        min_duration_s=min_duration_s,
        height_threshold=height_threshold,
        asset_cfg=asset_cfg,
    ).float().unsqueeze(-1)


def ref_base_pos_z(
    env: "ManagerBasedAnimationEnv",
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    """Reference root height sequence from the animation manager."""
    animation_term: "AnimationTerm" = env.animation_manager.get_term(animation)
    ref_root_pos_w = animation_term.get_root_pos_w()  # (num_envs, num_steps, 3)
    ref_root_height = ref_root_pos_w[:, :, 2:3]  # keep dim for consistency with other ref terms
    if flatten_steps_dim:
        return ref_root_height.reshape(env.num_envs, -1)
    return ref_root_height
