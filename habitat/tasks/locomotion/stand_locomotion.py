from collections import OrderedDict

import cv2
import numpy as np

from habitat import Config
from habitat.tasks.locomotion.locomotion_base_env import LocomotionRLEnv
from habitat.utils.geometry_utils import wrap_heading
from habitat_baselines.common.baseline_registry import baseline_registry


class LocomotionStandMixin:
    """When inheriting this Mixin, make sure that the RLEnv's init is called
    before the init of this class"""

    def __init__(self):
        # The following represents a standing pose:
        self.target_joint_positions = self.robot.standing_pose
        self.use_exp_mse = self.config.RL.use_exp_mse
        self.goal_height = 0.486

    def _task_reset(self):
        self.robot.reset()
        self.robot.prone()

    def _get_reward_terms(self, observations) -> np.array:
        reward_terms = OrderedDict()

        reward_terms["imitation_reward"] = (
            -wrap_heading(
                self.robot.joint_positions - self.target_joint_positions
            )
            ** 2
            / self.num_joints
        )

        # energy = self.robot.joint_torques * self.robot.joint_velocities
        # reward_terms["energy_reward"] = -np.abs(np.sum(energy, -1))
        self.named_rewards = reward_terms

        # Log accumulated reward info
        for k, v in reward_terms.items():
            if self.is_vector_env and v.ndim == 1:
                # Unsqueeze scalar reward terms
                reward_terms[k] = np.expand_dims(v, axis=1)
            self.accumulated_reward_info["cumul_" + k] += np.sum(
                reward_terms[k], -1
            )

        # Return just the values
        reward_values = [
            # Scalars are turned into 1-element vectors for concatenation
            np.array([v], dtype=np.float32) if isinstance(v, float) else v
            for v in reward_terms.values()
        ]
        axis = 1 if self.is_vector_env else 0
        reward_terms = np.concatenate(reward_values, axis=axis).astype(
            np.float32
        )

        return reward_terms

    def _baseline_policy(self):
        f"""Task-specific policy which produces actions that should maximize
        reward for debug purposes
        """
        deltas = self.target_joint_positions - self.robot.joint_positions
        deltas = np.clip(deltas, -self.max_rad_delta, self.max_rad_delta)
        deltas[abs(deltas) < self.success_thresh / 3] = 0
        return deltas

    def _get_success(self):
        return abs(self.robot.height - self.goal_height) < 0.05

    def image_text(self, img):
        imitation = self.named_rewards["imitation_reward"].sum()
        img = cv2.putText(
            img,  # numpy array on which text is written
            f"imitation: {imitation:.3f}",  # text
            (20, 90),  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            0.75,  # font size
            (255, 255, 255, 255),  # font color
            3,  # font stroke
        )

        energy = self.named_rewards["energy_reward"][0]
        img = cv2.putText(
            img,  # numpy array on which text is written
            f"energy reward: {energy:.1f}",  # text
            (20, 130),  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            0.75,  # font size
            (255, 255, 255, 255),  # font color
            3,  # font stroke
        )

        return img


@baseline_registry.register_env(name="LocomotionRLEnvStand")
class LocomotionRLEnvStand(LocomotionStandMixin, LocomotionRLEnv):
    def __init__(self, config: Config, render=False, *args, **kwargs):
        LocomotionRLEnv.__init__(self, config, render=render, *args, **kwargs)
        super().__init__()
