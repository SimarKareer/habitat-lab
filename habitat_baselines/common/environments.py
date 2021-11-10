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
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
import habitat_sim.utils.viz_utils as vut
import numpy as np
import os

import habitat
import habitat_sim
from habitat import Config, Dataset
from habitat.core.spaces import ActionSpace
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
import habitat_baselines.common.knobs_environment
import magnum as mn
import quaternion as qt
from habitat_sim.physics import JointMotorSettings
from copy import copy
import math
import torch
import cv2
from collections import defaultdict, OrderedDict
from habitat_baselines.common.aliengo import AlienGo
from habitat.utils.geometry_utils import wrap_heading


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)

@baseline_registry.register_env(name="EnergyLocomotionRLEnv")
class EnergyLocomotionRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, render=False, *args, **kwargs):
        super().__init__(config, render, args=args, kwargs=kwargs) #not sure this is correct

    def get_reward_terms(self, observations) -> np.array:
        reward_terms = OrderedDict()
        # add each reward term here
    
    def get_observations(self):
        return {
            "joint_pos": self.robot.joint_positions,
            "joint_vel": self.robot.joint_velocities,
            "euler_rot": self.robot.get_rpy(),
            "feet_contact": self.robot.get_feet_contacts(),
        }

@baseline_registry.register_env(name="LocomotionRLEnv")
class LocomotionRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, render=False, *args, **kwargs):
        self.config = config
        self.sim_config = config.TASK_CONFIG.SIMULATOR
        self.task_config = config.TASK_CONFIG.TASK

        # Create action space
        self.num_joints = config.TASK_CONFIG.TASK.ACTION.NUM_JOINTS
        self.max_rad_delta = np.deg2rad(
            config.TASK_CONFIG.TASK.ACTION.MAX_DEGREE_DELTA
        )
        self.action_space = ActionSpace(
            {
                "joint_deltas": spaces.Box(
                    low=-self.max_rad_delta,
                    high=self.max_rad_delta,
                    shape=(self.num_joints,),
                    dtype=np.float32,
                )
            }
        )

        # Create observation space
        self.observation_space = spaces.Dict(
            {
                "joint_pos": spaces.Box(
                    low=-np.pi,
                    high=np.pi,
                    shape=(self.num_joints,),
                    dtype=np.float32,
                ),
                "joint_vel": spaces.Box(
                    low=-np.pi,
                    high=np.pi,
                    shape=(self.num_joints,),
                    dtype=np.float32,
                ),
                "euler_rot": spaces.Box(
                    low=-np.pi,
                    high=np.pi,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "feet_contact": spaces.Box(
                    low=0,
                    high=1,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

        # Create simulator
        self._sim = self.create_sim()

        # Place agent (for now, this is just a camera)
        self._place_agent()

        # Debug
        self.video_dir = config.VIDEO_DIR

        # Load the robot into the sim
        self.fixed_base = self.config.TASK_CONFIG.DEBUG.FIXED_BASE
        ao_mgr = self._sim.get_articulated_object_manager()
        robot_id = ao_mgr.add_articulated_object_from_urdf(
            self.sim_config.ROBOT_URDF, fixed_base=self.fixed_base
        )
        self.robot = AlienGo(
            robot_id, self._sim, self.fixed_base, self.task_config
        )

        # The following represents a standing pose:
        self.target_joint_positions = np.array([0, 0.432, -0.77] * 4)

        # Initialize attributes
        self.render = render
        self.viz_buffer = []
        self.num_steps = 0
        self._max_episode_steps = (
            config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        )
        self.success_thresh = np.deg2rad(
            config.TASK_CONFIG.ENVIRONMENT.SUCCESS_THRESH
        )
        self.settle_time = config.TASK_CONFIG.SIMULATOR.SETTLE_TIME
        self.use_exp_mse = self.config.RL.use_exp_mse
        self.render_episode = False
        self.last_joint_pos = self.robot.robot_id.joint_positions
        self.sim_hz = self.sim_config.SIM_HZ
        self.ctrl_hz = self.sim_config.CTRL_HZ
        self.number_of_episodes = int(1e24)  # needed b/c habitat wants it
        self.goal_height = 0.486
        self.accumulated_reward_info = defaultdict(float)

    def reset(self):
        self.robot.reset()
        self.robot.prone()

        # Let robot settle on the ground
        self._sim.step_physics(self.settle_time)

        self.viz_buffer = []
        self.num_steps = 0
        self.accumulated_reward_info = defaultdict(float)

        return self.get_observations()

    def get_observations(self):
        return {
            "joint_pos": self.robot.joint_positions,
            "joint_vel": self.robot.joint_velocities,
            "euler_rot": self.robot.get_rpy(),
            "feet_contact": self.robot.get_feet_contacts(),
        }

    def get_reward_terms(self, observations) -> np.array:
        reward_terms = OrderedDict()

        # # Penalize roll and pitch
        # reward_terms["roll_pitch_penalty"] = -np.abs(
        #     observations["euler_rot"][:2]
        # )

        # # Penalize non-contacting feet
        # reward_terms["contact_feet_penalty"] = observations["feet_contact"] - 1

        # # Penalize deviation from desired height
        # reward_terms["height_penalty"] = [
        #     -np.abs(self.robot.height - self.goal_height)
        # ]

        # Penalize deviations from nominal standing pose (MSE)
        reward_terms["imitation_reward"] = (
            -wrap_heading(
                self.robot.joint_positions - self.target_joint_positions
            )
            ** 2
            / self.num_joints
        )

        # Log accumulated reward info
        for k, v in reward_terms.items():
            self.accumulated_reward_info["cumul_" + k] += np.sum(v)

        # Return just the values
        reward_terms = np.concatenate(list(reward_terms.values())).astype(
            np.float32
        )

        return reward_terms

    def step(self, action, action_args, *args, **kwargs):
        """Updates robot with given actions and calls physics sim step"""
        deltas = action_args["joint_deltas"]

        # Clip actions and scale
        deltas = np.clip(deltas, -1.0, 1.0) * self.max_rad_delta

        """ Hack below! Overwrite actions to always be successful """
        USE_HACK = False
        if USE_HACK:
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
        """ End of hack. """

        # Update current state
        self.robot.add_jms_pos(deltas)
        self._sim.step_physics(1.0 / self.sim_hz)
        


        # Return observations (error for each knob)
        observations = self.get_observations()

        # Get reward
        reward_terms = self.get_reward_terms(observations)
        reward = sum(reward_terms)

        # Text on Screen
        if self.render:
            self.viz_buffer.append(self._sim.get_sensor_observations())
            # print(self.viz_buffer[-1])
            self.viz_buffer[-1]["rgba_camera"] = cv2.putText(
                self.viz_buffer[-1]["rgba_camera"], #numpy array on which text is written
                f"Reward: {reward}", #text
                (20, 20), #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                1, #font size
                (256, 0, 0, 255), #font color
                3#font stroke
            )

        # Check termination conditions
        success = abs(self.robot.height - self.goal_height) < 0.05
        self.num_steps += 1
        done = self.num_steps == self._max_episode_steps
        if done and self.render:
            vut.make_video(
                self.viz_buffer,
                "rgba_camera",
                "color",
                os.path.join(self.video_dir, "vid{rand}.mp4".format(rand=np.random.randint(0, 1e6))),
                open_vid=False,
                fps=self.sim_hz
            )

        # Populate info for tensorboard
        info = {
            "success": 1.0 if success else 0.0,
            "height": self.robot.height,
            "reward_terms": reward_terms,
        }

        # Add info about how much of each reward component has accumulated
        info.update(self.accumulated_reward_info)

        return observations, reward, done, info

    def _place_agent(self):
        """Places our camera agent in a spot it can see the robot"""
        # place our agent in the scene
        agent_state = habitat_sim.AgentState()
        agent_state.position = [-1.1, 0.5, 1.1]
        agent_state.rotation *= qt.from_euler_angles(0.0, np.deg2rad(-40), 0.0)
        agent = self._sim.initialize_agent(0, agent_state)

        return agent.scene_node.transformation_matrix()

    def create_sim(self):
        # Simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.gpu_device_id = self.config.SIMULATOR_GPU_ID
        backend_cfg.enable_physics = True

        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        if self.render:
            # Sensor configurations
            rgb_camera = habitat_sim.CameraSensorSpec()
            rgb_camera.uuid = "rgba_camera"
            rgb_camera.sensor_type = habitat_sim.SensorType.COLOR

            # Make camera res super low if we're not rendering
            rgb_camera.resolution = [540, 720]
            rgb_camera.position = [0.0, 0.0, 0.0]
            rgb_camera.orientation = [0.0, 0.0, 0.0]

            agent_cfg.sensor_specifications = [rgb_camera]
        else:
            agent_cfg.sensor_specifications = []

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

        # Create simulator from configuration
        sim = habitat_sim.Simulator(cfg)

        # Load the floor
        obj_template_mgr = sim.get_object_template_manager()
        cube_handle = obj_template_mgr.get_template_handles("cubeSolid")[0]
        cube_template_cpy = obj_template_mgr.get_template_by_handle(
            cube_handle
        )
        cube_template_cpy.scale = np.array([5.0, 0.2, 5.0])
        cube_template_cpy.friction_coefficient = 0.3
        obj_template_mgr.register_template(cube_template_cpy)
        rigid_obj_mgr = sim.get_rigid_object_manager()
        ground_plane = rigid_obj_mgr.add_object_by_template_handle(cube_handle)
        ground_plane.translation = [0.0, -0.2, 0.0]
        ground_plane.motion_type = habitat_sim.physics.MotionType.STATIC

        return sim

    def close(self):
        pass

    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)


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
