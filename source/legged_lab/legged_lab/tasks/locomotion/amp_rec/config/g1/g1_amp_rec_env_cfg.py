import os
import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp_rec.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.tasks.locomotion.amp_rec.amp_rec_env_cfg import LocomotionAmpRecEnvCfg


KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4


@configclass
class G1AmpRecRewards:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp_when_upright,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "fallen_min_duration_s": 0.25,
            "fallen_height_threshold": 0.32,
            "fall_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    upright_recovery = RewTerm(
        func=mdp.upright_recovery_reward,
        weight=1.2,
        params={
            "fallen_min_duration_s": 0.25,
            "fallen_height_threshold": 0.32,
            "up_asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    root_height_recovery = RewTerm(
        func=mdp.root_height_recovery_reward,
        weight=1.0,
        params={
            "target_height": 0.75,
            "std": 0.2,
            "fallen_min_duration_s": 0.25,
            "fallen_height_threshold": 0.32,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class G1AmpRecEnvCfg(LocomotionAmpRecEnvCfg):
    rewards: G1AmpRecRewards = G1AmpRecRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp_bvh_base"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "run1_subject2": 1.0,
            "walk1_subject1": 1.0,
            "fallAndGetUp1_subject1": 1.0,
        }

        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.history_length = AMP_NUM_STEPS

        torso_cfg = SceneEntityCfg("robot", body_names="torso_link")

        # Remove following observations from policy input
        # self.observations.policy.torso_contacts.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link")}
        # self.observations.policy.knee_contacts.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_link")}
        # self.observations.policy.arm_contacts.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_shoulder_.*|.*_elbow_.*|.*_wrist_.*")}
        # self.observations.policy.fallen_fast.params = {
        #     "asset_cfg": torso_cfg,
        #     "min_duration_s": 0.25,
        #     "height_threshold": 0.32,
        # }

        self.observations.critic.torso_contacts.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link")}
        self.observations.critic.knee_contacts.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_link")}
        self.observations.critic.arm_contacts.params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_shoulder_.*|.*_elbow_.*|.*_wrist_.*")}
        self.observations.critic.fallen_fast.params = {
            "asset_cfg": torso_cfg,
            "min_duration_s": 0.25,
            "height_threshold": 0.32,
        }
        self.observations.critic.fallen_persistent.params = {
            "asset_cfg": torso_cfg,
            "min_duration_s": 2.0,
            "height_threshold": 0.32,
        }

        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_height.params["animation"] = ANIMATION_TERM_NAME

        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1,
            "fall_probability": 0.3,
            "lie_height": 0.24,
            "crouch_height": 0.45,
        }

        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None

        self.terminations.base_contact = None
        self.terminations.base_height = None


@configclass
class G1AmpRecEnvCfg_PLAY(G1AmpRecEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
