from collections import OrderedDict
import cv2
from habitat import Config
from habitat.tasks.locomotion.locomotion_base_env import LocomotionRLEnv
from habitat.utils.geometry_utils import wrap_heading
from habitat_baselines.common.baseline_registry import baseline_registry
import numpy as np


@baseline_registry.register_env(name="LocomotionRLEnvStand")
class LocomotionRLEnvStand(LocomotionRLEnv):
    def __init__(self, config: Config, render=False, *args, **kwargs):
        super().__init__(config, render=render, args=args, kwargs=kwargs)
        # The following represents a standing pose:
        self.target_joint_positions = np.array([0, 0.8, -1.5] * 4)
        self.use_exp_mse = self.config.RL.use_exp_mse
        self.goal_height = 0.486

    def _task_reset(self):
        self.robot.reset()
        self.robot.prone()

    def _baseline_policy(self):
        f""" Task-specific policy which produces actions that should maximize
        reward for debug purposes
        """
        print(
            "Target: ",
            self.target_joint_positions,
            "Current: ",
            self.robot.joint_positions,
        )
        deltas = []
        for i, j in zip(
            self.robot.joint_positions, self.target_joint_positions
        ):
            err = i - j
            # Be more precise than necessary
            if abs(err) > self.success_thresh / 3:
                # Flip direction based on error
                coeff = 1 if err < 0 else -1
                deltas.append(coeff * min(self.max_rad_delta, abs(err)))
            else:
                deltas.append(0)
        deltas = np.array(deltas, dtype=np.float32)
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

    def _get_reward_terms(self, observations) -> np.array:
        reward_terms = OrderedDict()

        reward_terms["imitation_reward"] = (
            -wrap_heading(
                self.robot.joint_positions - self.target_joint_positions
            )
            ** 2
            / self.num_joints
        )

        reward_terms["energy_reward"] = np.array(
            [
                -np.abs(
                    self.robot.joint_torques().dot(self.robot.joint_velocities)
                )
            ]
        )
        self.named_rewards = reward_terms

        # Log accumulated reward info
        for k, v in reward_terms.items():
            self.accumulated_reward_info["cumul_" + k] += np.sum(v)

        # Return just the values
        reward_terms = np.concatenate(list(reward_terms.values())).astype(
            np.float32
        )

        return reward_terms
