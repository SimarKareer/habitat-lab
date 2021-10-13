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


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)

@baseline_registry.register_env(name="LocomotionRLEnv")
class LocomotionRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self.sim_config = config.TASK_CONFIG.SIMULATOR
        self.task_config = config.TASK_CONFIG.TASK
        # Set up actions
        self.num_joints = config.TASK_CONFIG.TASK.ACTION.NUM_JOINTS
        self.action_space = ActionSpace({
            "joint_targets": spaces.Box(
                low=np.ones(self.num_joints) * np.deg2rad(-1),
                high=np.ones(self.num_joints) * np.deg2rad(1),
                dtype=np.float32,
            )}
        )

        # Create sim
        cfg = self._make_configuration()
        self._sim = habitat_sim.Simulator(cfg)

        # Place agent
        self._place_agent()

        # Load the robot into the sim
        ao_mgr = self._sim.get_articulated_object_manager()
        self.robot_id = ao_mgr.add_articulated_object_from_urdf(
            self.sim_config.ROBOT_URDF, fixed_base=False
        )

        # Load the floor
        obj_template_mgr = self._sim.get_object_template_manager()
        cube_handle = obj_template_mgr.get_template_handles("cubeSolid")[0]
        cube_template_cpy = obj_template_mgr.get_template_by_handle(cube_handle)
        cube_template_cpy.scale = np.array([5.0, 0.2, 5.0])
        obj_template_mgr.register_template(cube_template_cpy)
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        ground_plane = rigid_obj_mgr.add_object_by_template_handle(cube_handle)
        ground_plane.translation = [0.0, -0.2, 0.0]
        ground_plane.motion_type = habitat_sim.physics.MotionType.STATIC

        # Initialize attributes
        self.viz_buffer = []
        self.episode_steps = 0
        self.render_episode = False
        self.last_joint_pos = self.robot_id.joint_positions
        self.sim_hz = self.sim_config.SIM_HZ
        self.ctrl_hz = self.sim_config.CTRL_HZ
        self.jmsIdxToJoint = {0: "FL_hip", 1: "FL_thigh", 2: "FL_calf", 3: "FR_hip", 4: "FR_thigh", 5: "FR_calf", 6: "RL_hip", 7: "RL_thigh", 8: "RL_calf", 9: "RR_hip", 10: "RR_thigh", 11: "RR_calf"}
        # self._sim.set_gravity([0., 0., 0.])
        # super().__init__(self._core_env_config, dataset)

    def _jms_copy(self, jms):
        """ Returns a deep copy of a jms
            
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
        """ Returns a new jms with default settings at a given position
            
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

    def _reset_prone(self):
        """Resets robot in a legs bent stance
        """
        calfDofs = [2, 5, 8, 11]
        pose = self.robot_id.joint_positions
        for dof in calfDofs:
            pose[dof] = self.task_config.START.CALF #second joint
            pose[dof - 1] = self.task_config.START.THIGH #first joint
        self.robot_id.joint_positions = pose

        pose = self.robot_id.joint_positions
        for dof in calfDofs:
            pose[dof] = -1.8 #second joint
            pose[dof - 1] = 1.0 #first joint
        self.target_joint_positions = pose
        self.robot_id.joint_positions = pose
        # Roll robot 90 deg
        base_transform = mn.Matrix4.rotation(
            mn.Rad(np.deg2rad(-90)), mn.Vector3(1.0, 0.0, 0.0)
        )
        # Position above center of platform
        base_transform.translation = mn.Vector3(0.0, 0.3, 0.0)
        self.robot_id.transformation = base_transform

    def _reset_robot(self):
        """ Resets robot's position and jms settings
        
        """
        self._reset_prone()
        base_jms = self._new_jms(0)
        for i in range(12):
            self.robot_id.update_joint_motor(i, base_jms)
        
        self._set_joint_type_pos("thigh", self.task_config.START.THIGH)
        self._set_joint_type_pos("calf", self.task_config.START.CALF)
    
    def _set_joint_type_pos(self, joint_type, joint_pos):
        """ Set's all joints of a given type to a given position

            Args:
                joint_type: type of joint ie hip, thigh or calf
                joint_pos: position to set these joints to
        """
        for idx, joint_name in self.jmsIdxToJoint.items():
            if joint_type in joint_name:
                # print(f"Updating {joint_name}({idx}) to {joint_pos}".format(joint_name, idx, joint_pos))
                newjms = self._new_jms(joint_pos)
                self.robot_id.update_joint_motor(idx, newjms)

    def _place_agent(self):
        """ Places our camera agent in a spot it can see the robot
        """
        # place our agent in the scene
        agent_state = habitat_sim.AgentState()
        agent_state.position = [-1.1, 0.5, 1.1]
        agent_state.rotation *= qt.from_euler_angles(
            0.0, np.deg2rad(-40), 0.0
        )
        agent = self._sim.initialize_agent(0, agent_state)

        return agent.scene_node.transformation_matrix()

    def _make_configuration(self):
        # simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.enable_physics = True

        # sensor configurations
        rgb_camera = habitat_sim.CameraSensorSpec()
        rgb_camera.uuid = "rgba_camera"
        rgb_camera.sensor_type = habitat_sim.SensorType.COLOR
        rgb_camera.resolution = [540, 720]
        rgb_camera.position = [0.0, 0.0, 0.0]
        rgb_camera.orientation = [0.0, 0.0, 0.0]

        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_camera]

        return habitat_sim.Configuration(backend_cfg, [agent_cfg])

    def _simulate(self, steps=1, render=False):
        """ Runs physics simulator for given number of steps and returns sensor observations
            Args:
                steps: number of steps to take at 1/self.sim_hz timesteps
                render: whether or not observations should include the camera images
            
            Returns:
                observations: an array indexed by time of dictionaries where each dictionary maps sensor name to value 
                ex) [{sensor name: sensorValue_0}, {sensor name: sensorValue_1}]
        """
        # simulate dt seconds at 60Hz to the nearest fixed timestep
        observations = []
        for _ in range(steps):
            self._sim.step_physics(1.0 / self.sim_hz)
            vis = {}
            if render:
                vis = self._sim.get_sensor_observations()
            obs = self._read_sensors() #NOTE: It's important that this is run every time step physics is run so that the join velocity stays correct
            obs.update(vis)
            observations.append(obs)

        return observations
    
    def _add_jms_pos(self, joint_pos):
        """
            Updates existing joint positions by adding each position in array of 
            joint_positions
            Args
                joint_pos: array of delta joint positions
        """
        for i, new_pos in enumerate(joint_pos):
            jms = self._jms_copy(self.robot_id.get_joint_motor_settings(i))
            jms.position_target = self._fix_heading(jms.position_target + new_pos)
            self.robot_id.update_joint_motor(i, jms)

    def _read_sensors(self):
        """ Returns sensor observation for joint positions and velocities
        """
        obs = {}
        obs["joint_pos"] = np.array(self.robot_id.joint_positions)
        obs["joint_vel"] = (np.array(self.robot_id.joint_positions) - np.array(self.last_joint_pos)) / (1 / self.sim_hz)
        obs["euler_rot"] = self._get_robot_rpy()
        self.last_joint_pos = self.robot_id.joint_positions
        return obs
    
    def _get_robot_rpy(self):
        """Given a numpy quaternion we'll return the roll pitch yaw
            
            Returns:
                rpy: tuple of roll, pitch yaw
        """
        quat = self.robot_id.rotation.normalized()
        undo_rot = mn.Quaternion(((np.sin(np.deg2rad(45)), 0.0, 0.0), np.cos(np.deg2rad(45)))).normalized()
        # print("quat: {quat}, undo_rot: {undo_rot}, mult: {mult}".format(quat=quat, undo_rot=undo_rot, mult=quat*undo_rot))
        quat = quat*undo_rot

        x, y, z = quat.vector
        w = quat.scalar

        roll, pitch, yaw = self._euler_from_quaternion(x, y, z, w)
        return roll, pitch, yaw
    
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
    
        return roll_x, -yaw_z, pitch_y # in radians
    
    def _fix_heading(self, heading):
        if heading < -np.pi:
            heading += 2 * np.pi
        elif heading >= np.pi:
            heading -= 2 * np.pi
        
        return heading

    def _step_reward(self, step_obs):
        joint_pos = np.array(step_obs["joint_pos"])
        target_joint_pos = np.array(self.target_joint_positions)
        mse = (joint_pos - target_joint_pos)**2
        mse = 0.5 * np.array(
            [self._fix_heading(j - t) ** 2 for j, t in zip(joint_pos, target_joint_pos)]
        ).mean()
        # print("MSE: ", mse)

        reward = -mse
        if (np.abs(step_obs["euler_rot"]) > np.deg2rad(60)).any():
            # print("TIP PENALTY INCURRED")
            reward -= self.task_config.REWARD.TIP_PENALTY
        
        return reward


    def reset(self, render_episode=False):
        """ Resets episode by resetting robot, visualization buffer, and allowing robot to fall to ground

            Args:
                render_epsiode: whether or not to render video for this episode
            
            Returns:
                None
        """
        self.render_episode = render_episode
        self._reset_robot()

        # Let the robot fall to the ground
        obs = self._simulate(steps=240, render=render_episode)
        print("INIT REWARD: ", self._step_reward(obs[0]))
        self.viz_buffer = obs
        self.episode_steps = 0

        observations = None

        return observations

    def step(self, *args, **kwargs):
        """ Updates robot with given actions and calls physics sim step
        """
        self.episode_steps += 1
        self._add_jms_pos(args[0]["action_args"]) #NOTE: this seems a bit weird
        #NOTE: SIM_HZ/CTRL_HZ allows us to update motors at correct freq.
        assert int(self.sim_hz/self.ctrl_hz) == self.sim_hz/self.ctrl_hz, "sim_hz should be divis by ctrl_hz for simplicity"
        obs = self._simulate(steps=int(self.sim_hz/self.ctrl_hz), render=self.render_episode)


        # print("joint pos: ", obs["joint_pos"])
        # print("joint vel: ", obs["joint_vel"])
        # print("-"*100)
        self.viz_buffer += obs #+= if obs is array, append else
        step_obs = obs[-1]
        # print("Robot RYP: ", np.rad2deg(step_obs["euler_rot"]))

        done = self.episode_steps >= 500
        if done and self.render_episode:
            vut.make_video(
                self.viz_buffer,
                "rgba_camera",
                "color",
                "vid.mp4",
                open_vid=False,
            )
        # print(self.robot_id.joint_positions)

        info = {}
        reward = self._step_reward(step_obs)
        # print("REWARD: ", reward)
        return step_obs, reward, done, info
        # return super().step(*args, **kwargs)


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
