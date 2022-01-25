#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import numpy as np
import magnum as mn
import cv2
import os

import habitat
from habitat import Config, Dataset

from habitat.tasks.locomotion.locomotion_base_env import LocomotionRLEnv
from collections import defaultdict, OrderedDict

from habitat.utils.geometry_utils import wrap_heading
from habitat_baselines.common.baseline_registry import baseline_registry
import habitat_sim.utils.viz_utils as vut


@baseline_registry.register_env(name="LocomotionRLEnvEnergy")
class LocomotionRLEnvEnergy(LocomotionRLEnv):
    def __init__(self, config: Config, render=False, *args, **kwargs):
        super().__init__(config, render=render, args=args, kwargs=kwargs)
        self.target_velocity = np.array(
            [self.task_config.TASK.TARGET_VELOCITY, 0, 0]
        )
        self.target_forward_velocity = self.task_config.TASK.TARGET_VELOCITY
        self.rewards = self.task_config.TASK.REWARD
        self.time_step = self._sim.get_physics_time_step()

        print("time step: ", self.time_step)

    def _task_reset(self):
        self.robot.reset()
        self.robot.stand()

    def _get_success(self):
        return abs(self.robot.forward_velocity - self.target_velocity[0]) < 0.1
        # self.robot_id.root_angular_velocity = mn.Vector3(0.0, 0.0, 0.0)

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

        img = vut.append_text_to_image(img, lines)
        return img

    def _get_reward_terms(self, observations) -> np.array:
        reward_terms = OrderedDict()

        reward_terms["forward_velocity_reward"] = np.array(
            [
                -self.rewards.a2
                * np.abs(
                    self.robot.forward_velocity - self.target_forward_velocity
                )
            ]
        )
        reward_terms["side_velocity_reward"] = np.array(
            [-self.robot.side_velocity ** 2]
        )
        reward_terms["angular_velocity_reward"] = np.array(
            [-self.robot.robot_id.root_angular_velocity[1] ** 2]
        )

        # r, p, y = np.abs(self.robot.get_rpy())
        # reward_terms["balance_reward"] = np.array(
        #     [
        #         self.rewards.wux * r
        #         + self.rewards.wuy * p
        #         + self.rewards.wuz * y
        #     ]
        # )

        # print("POS: ", self.robot.position)
        # _, _, zpos = self.robot.position
        # reward_terms["sideways_reward"] = np.array([-np.abs(zpos)])
        # reward_terms["alive"] = np.array([self.rewards.alive])
        # reward_terms["energy"] = np.array(
        #     [
        #         -self.rewards.we
        #         * np.linalg.norm(self.robot.robot_id.joint_forces)
        #     ]
        # )
        # print("REWARD TERMS: ", reward_terms)

        # print("TORQUE: ", self.robot.joint_torques)
        # print("VEL: ", self.robot.joint_velocities)

        reward_terms["energy_reward"] = np.array(
            [
                -np.abs(
                    self.robot.joint_torques().dot(self.robot.joint_velocities)
                )
            ]
        )

        reward_terms["alive_reward"] = np.array(
            [self.rewards.alive * self.target_forward_velocity]
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

    def should_end(self):
        roll, pitch = self.robot.get_rp()
        # print("Terminated Episode: ", self.robot.height, roll, pitch)
        # return False
        return (
            self.robot.height < 0.28 or abs(roll) > 0.4 or abs(pitch) > 0.2
        )  # REMEMBER: uncomment

    def add_force(self, fx, fy, fz, link=0):
        self.robot.robot_id.add_link_force(link, mn.Vector3(fx, fy, fz))

    # def force_adjustment(self, kp, kd):
    #     forward_force = kp * (self.target_forward_velocity - self.robot.forward_velocity) #TODO add derivative term as well
    #     self.robot.robot_id.add_link_force(0, mn.Vector3(forward_force, 0, 0))
    #     side_force = kp * (-self.robot.position[2]) + kd * (-self.robot.side_velocity)
    #     self.robot.robot_id.add_link_force(0, mn.Vector3(0, 0, side_force))


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
        f""" Task Specific policy which produces actions that should maximize reward for debug purposes 
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


@baseline_registry.register_env(name="RearrangeRLEnv")
class RearrangeRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)
