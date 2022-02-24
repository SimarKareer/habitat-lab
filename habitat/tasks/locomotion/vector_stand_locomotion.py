import numpy as np

from habitat import Config
from habitat.tasks.locomotion.locomotion_vector_env import (
    LocomotionVectorRLEnv,
)
from habitat.tasks.locomotion.stand_locomotion import LocomotionStandMixin
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="LocomotionVectorRLEnvStand")
class LocomotionVectorRLEnvStand(LocomotionStandMixin, LocomotionVectorRLEnv):
    def __init__(self, config: Config, render=False, *args, **kwargs):
        LocomotionVectorRLEnv.__init__(
            self, config, render=render, *args, **kwargs
        )
        super().__init__()
        # Foot contact is not supported currently for vector env
        self.observation_space.spaces.pop("feet_contact")

    def reset_done_envs(self, done, observations):
        # Reset robots that are done
        done_robot_inds = np.where(done == 1.0)[0]
        self.robot.reset(robot_inds=done_robot_inds)
        self.robot.prone(robot_inds=done_robot_inds)

        # Replace terminal observations with initial state observations
        for robot_idx in done_robot_inds:
            robot = self.robot.robots[robot_idx]
            observations["joint_pos"][robot_idx] = robot.joint_positions
            observations["joint_vel"][robot_idx] = robot.joint_velocities
            observations["euler_rot"][robot_idx] = robot.rp

            # Clear out accumulated reward terms
            for k in self.accumulated_reward_info.keys():
                self.accumulated_reward_info[k][robot_idx] = 0.0

        return observations

    def _get_observations(self):
        """Foot contact is not supported currently for vector env"""
        return {
            "joint_pos": self.robot.joint_positions,
            "joint_vel": self.robot.joint_velocities,
            "euler_rot": self.robot.rp,
        }

    def get_done_info(self, reward_terms, step_render=False):
        done, info = super().get_done_info(reward_terms, step_render)
        info["height"] = self.robot.height
        return done, info
