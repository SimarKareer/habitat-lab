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
import magnum as mn
import quaternion as qt
from habitat_sim.physics import JointMotorSettings
from copy import copy
import math
import torch


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


class AlienGo:
    def __init__(self, robot_id, sim, fixed_base, task_config):
        self.robot_id = robot_id
        self._sim = sim
        self.fixed_base = fixed_base
        self.task_config = task_config

        self.jmsIdxToJoint = [
            "FL_hip",
            "FL_thigh",
            "FL_calf",
            "FR_hip",
            "FR_thigh",
            "FR_calf",
            "RL_hip",
            "RL_thigh",
            "RL_calf",
            "RR_hip",
            "RR_thigh",
            "RR_calf",
        ]

        # joint position limits
        self.joint_limits_lower = np.array(
            [-0.1, -np.pi / 3, -5 / 6 * np.pi] * 4
        )
        self.joint_limits_upper = np.array([0.1, np.pi / 2.1, -np.pi / 4] * 4)

    @property
    def joint_positions(self) -> np.ndarray:
        return np.array(self.robot_id.joint_positions, dtype=np.float32)

    def set_joint_positions(self, pose):
        self.robot_id.joint_positions = self._fix_heading(pose)


    def _jms_copy(self, jms):
        """Returns a deep copy of a jms

        Args:
            jms: the jms to copy
        """
        return JointMotorSettings(
            jms.position_target,
            jms.position_gain,
            jms.velocity_target,
            jms.velocity_gain,
            jms.max_impulse,
        )

    def _new_jms(self, pos):
        """Returns a new jms with default settings at a given position

        Args:
            pos: the new position to set to
        """
        return JointMotorSettings(
            pos,  # position_target
            0.6,  # position_gain
            0.0,  # velocity_target
            1.5,  # velocity_gain
            1.0,  # max_impulse
        )

    def prone(self):
        # Bend legs
        calfDofs = [2, 5, 8, 11]
        pose = np.zeros(12, dtype=np.float32)
        for dof in calfDofs:
            pose[dof] = self.task_config.START.CALF  # second joint
            pose[dof - 1] = self.task_config.START.THIGH  # first joint
        self.robot_id.joint_positions = pose

    def reset(self):
        # Zero out the link and root velocities
        self.robot_id.clear_joint_states()
        self.robot_id.root_angular_velocity = mn.Vector3(0.0, 0.0, 0.0)
        self.robot_id.root_linear_velocity = mn.Vector3(0.0, 0.0, 0.0)

        # Roll robot 90 deg
        base_transform = mn.Matrix4.rotation(
            mn.Rad(np.deg2rad(-90)), mn.Vector3(1.0, 0.0, 0.0)
        )

        # Position above center of platform
        base_transform.translation = (
            mn.Vector3(0.0, 0.8, 0.0)
            if self.fixed_base
            else mn.Vector3(0.0, 0.3, 0.0)
        )
        self.robot_id.transformation = base_transform

        # Make robot prone
        self.prone()

        # Reset all jms
        base_jms = self._new_jms(0)
        for i in range(12):
            self.robot_id.update_joint_motor(i, base_jms)

        self._set_joint_type_pos("thigh", self.task_config.START.THIGH)
        self._set_joint_type_pos("calf", self.task_config.START.CALF)

    def _set_joint_type_pos(self, joint_type, joint_pos):
        """Sets all joints of a given type to a given position

        Args:
            joint_type: type of joint ie hip, thigh or calf
            joint_pos: position to set these joints to
        """
        for idx, joint_name in enumerate(self.jmsIdxToJoint):
            if joint_type in joint_name:
                newjms = self._new_jms(joint_pos)
                self.robot_id.update_joint_motor(idx, newjms)

    def _add_jms_pos(self, joint_pos):
        """
        Updates existing joint positions by adding each position in array of
        joint_positions
        Args
            joint_pos: array of delta joint positions
        """
        for i, new_pos in enumerate(joint_pos):
            jms = self._jms_copy(self.robot_id.get_joint_motor_settings(i))
            temp = jms.position_target + new_pos
            pos_target = np.clip(
                temp, self.joint_limits_lower[i], self.joint_limits_upper[i]
            )
            jms.position_target = self._fix_heading(pos_target)
            self.robot_id.update_joint_motor(i, jms)

    def _get_robot_rpy(self):
        """Given a numpy quaternion we'll return the roll pitch yaw

        Returns:
            rpy: tuple of roll, pitch yaw
        """
        quat = self.robot_id.rotation.normalized()
        undo_rot = mn.Quaternion(
            ((np.sin(np.deg2rad(45)), 0.0, 0.0), np.cos(np.deg2rad(45)))
        ).normalized()
        quat = quat * undo_rot

        x, y, z = quat.vector
        w = quat.scalar

        roll, pitch, yaw = self._euler_from_quaternion(x, y, z, w)
        rpy = self._fix_heading(np.array([roll, pitch, yaw]))
        return rpy

    def _euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, yaw, pitch)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, -yaw_z, pitch_y  # in radians

    def _fix_heading(self, headings):
        return (headings + np.pi) % (2 * np.pi) - np.pi

def fix_heading(headings):
    return (headings + np.pi) % (2 * np.pi) - np.pi

@baseline_registry.register_env(name="LocomotionRLEnv")
class LocomotionRLEnv(habitat.RLEnv):
    def __init__(
        self,
        config: Config,
        *args,
        **kwargs
    ):
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
                "joint_targets": spaces.Box(
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
            }
        )

        # Create simulator
        self._sim = self.create_sim()

        # Place agent (for now, this is just a camera)
        self._place_agent()

        # Load the robot into the sim
        self.fixed_base = True
        ao_mgr = self._sim.get_articulated_object_manager()
        robot_id = ao_mgr.add_articulated_object_from_urdf(
            self.sim_config.ROBOT_URDF, fixed_base=self.fixed_base
        )
        self.robot = AlienGo(
            robot_id, self._sim, self.fixed_base, self.task_config
        )

        # Initialize attributes
        self.viz_buffer = []
        self.num_steps = 0
        self._max_episode_steps = (
            config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        )
        self.render_episode = False
        self.last_joint_pos = self.robot.robot_id.joint_positions
        self.sim_hz = self.sim_config.SIM_HZ
        self.ctrl_hz = self.sim_config.CTRL_HZ
        self.number_of_episodes = int(1e24)  # needed b/c habitat wants it

    def create_sim(self):
        # Simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.enable_physics = True

        # Sensor configurations
        rgb_camera = habitat_sim.CameraSensorSpec()
        rgb_camera.uuid = "rgba_camera"
        rgb_camera.sensor_type = habitat_sim.SensorType.COLOR

        # Make camera res super low if we're not rendering
        # rgb_camera.resolution = [540, 720]
        rgb_camera.resolution = [1, 1]

        rgb_camera.position = [0.0, 0.0, 0.0]
        rgb_camera.orientation = [0.0, 0.0, 0.0]

        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_camera]

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
        obj_template_mgr.register_template(cube_template_cpy)
        rigid_obj_mgr = sim.get_rigid_object_manager()
        ground_plane = rigid_obj_mgr.add_object_by_template_handle(cube_handle)
        ground_plane.translation = [0.0, -0.2, 0.0]
        ground_plane.motion_type = habitat_sim.physics.MotionType.STATIC

        return sim

    def _place_agent(self):
        """Places our camera agent in a spot it can see the robot"""
        # place our agent in the scene
        agent_state = habitat_sim.AgentState()
        agent_state.position = [-1.1, 0.5, 1.1]
        agent_state.rotation *= qt.from_euler_angles(0.0, np.deg2rad(-40), 0.0)
        agent = self._sim.initialize_agent(0, agent_state)

        return agent.scene_node.transformation_matrix()

    def close(self):
        pass

    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _simulate(self, steps=1, render=False):
        """Runs physics simulator for given number of steps and returns sensor
        observations

        Args:
            steps: number of steps to take at 1/self.sim_hz timesteps
            render: whether or not observations should include camera images

        Returns:
            observations: an array indexed by time of dictionaries where each
            dictionary maps sensor name to value
            ex) [{sensor name: sensorValue_0}, {sensor name: sensorValue_1}]
        """
        # simulate dt seconds at 60Hz to the nearest fixed timestep
        observations = []
        for _ in range(steps):
            self._sim.step_physics(1.0 / self.sim_hz)
            vis = {}
            if render:
                vis = self._sim.get_sensor_observations()
                self.viz_buffer.append(vis)
            obs = (
                self._read_sensors()
            )  # NOTE: It's important that this is run every time step physics
            # is run so that the join velocity stays correct
            # obs.update(vis)
            observations.append(obs)

        return observations

    def reset(self):
        """Resets episode by resetting robot, visualization buffer, and allowing robot to fall to ground

        Args:
            render_epsiode: whether or not to render video for this episode

        Returns:
            None
        """
        self.robot.reset()

        # Randomize joint positions to between -30 and 30
        self.robot.set_joint_positions(
            (
                np.random.rand(self.num_joints) * 2 - 1
            ) * np.deg2rad(30)
        )

        self.viz_buffer = []
        self.num_steps = 0
        self.cumul_reward = 0

        return {
            'joint_pos': self.robot.joint_positions
        }

    def step(self, action, *args, **kwargs):
        """Updates robot with given actions and calls physics sim step"""
        deltas = action["action_args"]["joint_deltas"]

        # Clip actions and scale
        deltas = np.clip(deltas, -1.0, 1.0) * self.max_rad_delta

        # '''Hack!'''
        # deltas = []
        # for i in self.robot.joint_positions:
        #     if i > np.deg2rad(3):
        #         deltas.append(-self.max_rad_delta)
        #     elif i < np.deg2rad(-3):
        #         deltas.append(-self.max_rad_delta)
        #     else:
        #         deltas.append(0)
        # deltas = np.array(deltas, dtype=np.float32)

        # Update current state
        self.robot.set_joint_positions(self.robot.joint_positions + deltas)

        # Return observations (error for each knob)
        observations = {
            'joint_pos': self.robot.joint_positions
        }

        # Penalize MSE
        reward = 0.5 * -np.mean(self.robot.joint_positions ** 2)

        # Check termination conditions
        success = (abs(self.robot.joint_positions) < np.deg2rad(3)).all()

        self.num_steps += 1
        done = success or self.num_steps == self._max_episode_steps

        self.cumul_reward += reward

        info = {
            'success': 1.0 if success else 0.0,
            'cumul_reward': self.cumul_reward
        }

        return observations, reward, done, info


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
