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
from collections import defaultdict, OrderedDict

from habitat.utils.geometry_utils import wrap_heading
from habitat_sim.utils.common import quat_from_magnum

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
    def height(self):
        # Translation is [y, z, x]
        return self.robot_id.rigid_state.translation[1]

    @property
    def translation(self):
        # Translation is [y, z, x]
        return self.robot_id.rigid_state.translation

    @property
    def joint_velocities(self) -> np.ndarray:
        return np.array(self.robot_id.joint_velocities, dtype=np.float32)

    @property
    def joint_positions(self) -> np.ndarray:
        return np.array(self.robot_id.joint_positions, dtype=np.float32)

    def set_joint_positions(self, pose):
        """This is kinematic! Not dynamic."""
        self.robot_id.joint_positions = wrap_heading(pose)

    def get_feet_contacts(self):
        """THIS ASSUMES THAT THERE IS ONLY ONE ROBOT IN THE SIM
        Returns np.array size 4, either 1s or 0s, for FL FR RL RR feet.
        """
        contacts = self._sim.get_physics_contact_points()
        contacting_feet = set()
        for c in contacts:
            for link in [c.link_id_a, c.link_id_b]:
                contacting_feet.add(self.robot_id.get_link_name(link))
        return np.array(
            [
                1 if foot in contacting_feet else 0
                for foot in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ]
        )

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

        # Snap joints kinematically
        self.robot_id.joint_positions = pose

        # Make motor controllers maintain this position
        for idx, p in enumerate(pose):
            self.robot_id.update_joint_motor(idx, self._new_jms(p))

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
                self.robot_id.update_joint_motor(idx, self._new_jms(joint_pos))

    def add_jms_pos(self, joint_pos):
        """
        Updates existing joint positions by adding each position in array of
        joint_positions
        Args
            joint_pos: array of delta joint positions
        """
        for i, new_pos in enumerate(joint_pos):
            jms = self._jms_copy(self.robot_id.get_joint_motor_settings(i))
            jms.position_target = np.clip(
                wrap_heading(jms.position_target + new_pos),
                self.joint_limits_lower[i],
                self.joint_limits_upper[i],
            )
            self.robot_id.update_joint_motor(i, jms)

    def get_rpy(self):
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
        rpy = wrap_heading(np.array([roll, pitch, yaw]))
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

        # Load the robot into the sim
        self.fixed_base = False
        ao_mgr = self._sim.get_articulated_object_manager()
        robot_id = ao_mgr.add_articulated_object_from_urdf(
            self.sim_config.ROBOT_URDF, fixed_base=self.fixed_base
        )
        self.robot = AlienGo(
            robot_id, self._sim, self.fixed_base, self.task_config
        )

        # Place agent (for now, this is just a camera)
        self._sim.initialize_agent(0, self.position_camera())

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

        # Penalize roll and pitch
        reward_terms["roll_pitch_penalty"] = -np.abs(
            observations["euler_rot"][:2]
        )

        # Penalize non-contacting feet
        reward_terms["contact_feet_penalty"] = observations["feet_contact"] - 1

        # Penalize deviation from desired height
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
        if self.render:
            self._sim.agents[0].set_state(self.position_camera())
            self.viz_buffer.append(self._sim.get_sensor_observations())

        # Return observations (error for each knob)
        observations = self.get_observations()

        # Get reward
        reward_terms = self.get_reward_terms(observations)
        reward = sum(reward_terms)

        # Check termination conditions
        success = abs(self.robot.height - self.goal_height) < 0.05
        self.num_steps += 1
        done = self.num_steps == self._max_episode_steps
        if done:
            vut.make_video(
                self.viz_buffer,
                "rgba_camera",
                "color",
                "test.mp4",
                fps=self.sim_hz,
                open_vid=False,
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

    def position_camera(self, move_with_robot=False):
        """Place camera at fixed offset from robot always pointed at robot"""
        offset = self.robot.translation
        if not move_with_robot:
            offset *= np.array([0.0, 1.0, 0.0])
        agent_position = offset + np.array([-1.1, 0.1, 1.1])
        agent_transformation = mn.Matrix4.look_at(
            agent_position,
            self.robot.translation,
            mn.Vector3(0.0, 1.0, 0.0),  # unit vector towards positive z axis
        )
        agent_state = habitat_sim.AgentState()
        agent_state.position = agent_position
        agent_mn_quaternion = mn.Quaternion.from_matrix(
            agent_transformation.rotation()
        )
        agent_state.rotation = quat_from_magnum(agent_mn_quaternion)

        return agent_state


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
