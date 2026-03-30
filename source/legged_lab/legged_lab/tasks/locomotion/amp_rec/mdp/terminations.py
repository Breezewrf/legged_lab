"""Termination terms for AMP-Rec environment."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_ground_contact_buffers(env: ManagerBasedRLEnv):
    """Initialize buffers for tracking ground contact duration."""
    if hasattr(env, "_amp_rec_ground_contact_steps"):
        return
    env._amp_rec_ground_contact_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._amp_rec_last_contact_update = -1


def _update_ground_contact_steps(
    env: ManagerBasedRLEnv,
    is_contact: torch.Tensor,
):
    """Update ground contact duration counter."""
    _ensure_ground_contact_buffers(env)

    current_step = int(env.common_step_counter)
    if env._amp_rec_last_contact_update == current_step:
        return

    # Reset per-env counters immediately after environment reset.
    just_reset = env.episode_length_buf == 0
    if torch.any(just_reset):
        env._amp_rec_ground_contact_steps[just_reset] = 0

    # Increment counter if in contact, reset otherwise
    env._amp_rec_ground_contact_steps = torch.where(
        is_contact,
        env._amp_rec_ground_contact_steps + 1,
        torch.zeros_like(env._amp_rec_ground_contact_steps),
    )
    env._amp_rec_last_contact_update = current_step


def sits_on_ground(
    env: ManagerBasedRLEnv,
    min_duration_s: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*"),
) -> torch.Tensor:
    """Terminate when robot sits on ground (sustained contact) for specified duration.
    
    Args:
        env: The environment.
        min_duration_s: Minimum duration (in seconds) robot must be in contact to trigger termination. Default: 1.0 second.
        sensor_cfg: Configuration for the contact sensor and body names to check.
    
    Returns:
        Boolean tensor indicating which environments should terminate (robot stayed in contact too long).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Check if any of the monitored bodies are in contact
    # current_contact_time > 0 means body is currently in contact
    contact_times = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    is_contact = torch.any(contact_times > 0, dim=1)
    
    # Update the contact duration tracking
    _update_ground_contact_steps(env, is_contact)
    
    # Calculate minimum steps needed
    min_steps = max(1, int(math.ceil(min_duration_s / env.step_dt)))
    
    # Terminate if sustained contact exceeds threshold
    return env._amp_rec_ground_contact_steps >= min_steps
