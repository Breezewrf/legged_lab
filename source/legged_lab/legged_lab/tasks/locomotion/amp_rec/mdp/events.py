from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from legged_lab.tasks.locomotion.deepmimic.mdp.events import reset_from_ref

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv


def _find_joint_ids(robot: Articulation, patterns: tuple[str, ...]) -> list[int]:
    ids = []
    for idx, name in enumerate(robot.joint_names):
        if any(p in name for p in patterns):
            ids.append(idx)
    return ids


def reset_from_ref_with_fall_probability(
    env: ManagerBasedAnimationEnv,
    env_ids: torch.Tensor,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.1,
    fall_probability: float = 0.3,
    lie_height: float = 0.24,
    crouch_height: float = 0.45,
):
    """Reset from reference and randomly initialize a subset into fallen/crouched states."""
    reset_from_ref(
        env=env,
        env_ids=env_ids,
        animation=animation,
        asset_cfg=asset_cfg,
        height_offset=height_offset,
    )

    if env_ids.numel() == 0 or fall_probability <= 0.0:
        return

    robot: Articulation = env.scene[asset_cfg.name]
    sample = torch.rand((env_ids.numel(),), device=env.device)
    fallen_env_ids = env_ids[sample < fall_probability]
    if fallen_env_ids.numel() == 0:
        return

    num_fallen = fallen_env_ids.numel()
    mode_selector = torch.rand((num_fallen,), device=env.device) < 0.5

    pose = torch.cat(
        [
            robot.data.root_pos_w[fallen_env_ids].clone(),
            robot.data.root_quat_w[fallen_env_ids].clone(),
        ],
        dim=-1,
    )
    vel = torch.zeros((num_fallen, 6), device=env.device, dtype=torch.float32)

    lie_idx = mode_selector
    crouch_idx = ~mode_selector

    if torch.any(lie_idx):
        lie_count = int(torch.sum(lie_idx).item())
        pose[lie_idx, 2] = lie_height
        signs = torch.where(
            torch.rand((lie_count,), device=env.device) > 0.5,
            torch.ones(lie_count, device=env.device),
            -torch.ones(lie_count, device=env.device),
        )
        # +/- 90 deg roll quaternion in wxyz convention.
        pose[lie_idx, 3] = 0.70710677
        pose[lie_idx, 4] = signs * 0.70710677
        pose[lie_idx, 5] = 0.0
        pose[lie_idx, 6] = 0.0

    if torch.any(crouch_idx):
        pose[crouch_idx, 2] = crouch_height

    robot.write_root_pose_to_sim(pose, env_ids=fallen_env_ids)
    robot.write_root_velocity_to_sim(vel, env_ids=fallen_env_ids)

    joint_pos = robot.data.default_joint_pos[fallen_env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    hip_ids = _find_joint_ids(robot, ("hip_pitch",))
    knee_ids = _find_joint_ids(robot, ("knee",))
    ankle_ids = _find_joint_ids(robot, ("ankle_pitch",))

    if torch.any(crouch_idx):
        crouch_rows = torch.nonzero(crouch_idx, as_tuple=False).squeeze(-1)
        if hip_ids:
            joint_pos[crouch_rows[:, None], hip_ids] = -0.9
        if knee_ids:
            joint_pos[crouch_rows[:, None], knee_ids] = 1.8
        if ankle_ids:
            joint_pos[crouch_rows[:, None], ankle_ids] = -0.9

    if torch.any(lie_idx):
        lie_rows = torch.nonzero(lie_idx, as_tuple=False).squeeze(-1)
        if hip_ids:
            joint_pos[lie_rows[:, None], hip_ids] = -0.35
        if knee_ids:
            joint_pos[lie_rows[:, None], knee_ids] = 0.7
        if ankle_ids:
            joint_pos[lie_rows[:, None], ankle_ids] = -0.35

    joint_limits = robot.data.soft_joint_pos_limits[fallen_env_ids]
    joint_pos = joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=fallen_env_ids)
