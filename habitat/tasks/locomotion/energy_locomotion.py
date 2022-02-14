from collections import OrderedDict

import magnum as mn
import numpy as np

from habitat import Config
from habitat.tasks.locomotion.locomotion_base_env import LocomotionRLEnv
from habitat.utils.visualizations.utils import overlay_text_to_image
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="LocomotionRLEnvEnergy")
class LocomotionRLEnvEnergy(LocomotionRLEnv):
    def __init__(self, config: Config, render=False, *args, **kwargs):
        super().__init__(config, render=render, args=args, kwargs=kwargs)
        self.target_velocity = np.array(
            [self.task_config.TASK.TARGET_VELOCITY, 0, 0]
        )
        self.target_forward_velocity = self.task_config.TASK.TARGET_VELOCITY
        self.rewards_cfg = self.task_config.TASK.REWARD
        self.terminate_on_tilt = self.task_config.TASK.TERMINATE_ON_TILT

    def _task_reset(self):
        self.robot.reset()
        self.robot.stand()

    def _get_success(self):
        return abs(self.robot.forward_velocity - self.target_velocity[0]) < 0.1

    def image_text(self, img):
        lines = []

        vxr = self.named_rewards["forward_velocity_reward"][0]
        vyr = self.named_rewards["side_velocity_reward"][0]
        lines.append(f"vx reward: {vxr:.3f} vy reward: {vyr:.3f}")

        vx = self.robot.forward_velocity
        vy = self.robot.side_velocity
        lines.append(f"vx: {vx:.3f} vy: {vy:.3f}")

        angr = self.named_rewards["angular_velocity_reward"][0]
        lines.append(f"angular vel reward: {angr:.3f}")

        wx, wy, wz = self.robot.robot_id.root_angular_velocity
        lines.append(f"angular vel: {wx:.1f}, {wy:.1f}, {wz:.1f}")

        energy = self.named_rewards["energy_reward"][0]
        lines.append(f"energy reward: {energy:.1f}")

        height = self.robot.height
        lines.append(f"Robot Height: {height:.2f}")

        img = overlay_text_to_image(img, lines)
        return img

    def _get_reward_terms(self, observations) -> np.array:
        reward_terms = OrderedDict()

        fwd_vel_error = np.abs(
            self.robot.forward_velocity - self.target_forward_velocity
        )
        reward_terms["forward_velocity_reward"] = np.array(
            [-self.rewards_cfg.a2 * fwd_vel_error]
        )
        reward_terms["side_velocity_reward"] = np.array(
            [-self.robot.side_velocity ** 2]
        )
        reward_terms["angular_velocity_reward"] = np.array(
            [-self.robot.robot_id.root_angular_velocity[1] ** 2]
        )

        energy = np.abs(
            self.robot.joint_torques.dot(self.robot.joint_velocities)
        )
        reward_terms["energy_reward"] = np.array([-energy])

        reward_terms["alive_reward"] = np.array(
            [self.rewards_cfg.alive * self.target_forward_velocity]
        )

        self.named_rewards = reward_terms

        # Log accumulated reward info
        for k, v in reward_terms.items():
            self.accumulated_reward_info["cumul_" + k] += np.sum(v)

        # Return just the values
        reward_terms = list(reward_terms.values())
        reward_terms = np.concatenate(reward_terms).astype(np.float32)

        return reward_terms

    def should_end(self):
        if not self.terminate_on_tilt:
            return False
        roll, pitch = self.robot.get_rp()
        return self.robot.height < 0.28 or abs(roll) > 0.4 or abs(pitch) > 0.2

    def add_force(self, fx, fy, fz, link=0):
        self.robot.robot_id.add_link_force(link, mn.Vector3(fx, fy, fz))
