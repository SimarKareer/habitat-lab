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
            self.sim_config.ROBOT_URDF, fixed_base=True
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
        # super().__init__(self._core_env_config, dataset)

    def _reset_robot(self):
        # target_pos = np.zeros(self.num_joints)
        
        # self.jms_list = [
        #     JointMotorSettings(
        #         pos,  # position_target
        #         0.09,  # position_gain
        #         0.0,  # velocity_target
        #         0.5,  # velocity_gain
        #         0.01,  # max_impulse
        #     )
        #     for pos in target_pos
        # ]
        
        # self.robot_id.create_all_motors(self.jms_list[0])
        # self._set_joint_type_pos("thigh", 0.9)
        # self._set_joint_type_pos("calf", -1.9)
        # self.robot_id.update_joint_motor(11, testJms)

        # Roll robot 90 deg
        base_transform = mn.Matrix4.rotation(
            mn.Rad(np.deg2rad(-90)), mn.Vector3(1.0, 0.0, 0.0)
        )
        # Position above center of platform
        base_transform.translation = mn.Vector3(0.0, 0.7, 0.0)
        self.robot_id.transformation = base_transform

        print("setting lay!!!")
        calfDofs = [2, 5, 8, 11]
        for dof in calfDofs:
            self.robot_id.joint_positions[dof] = -2.3 #second joint
            self.robot_id.joint_positions[dof - 1] = 1.3 #first joint

    
    def _set_joint_type_pos(self, joint_type, joint_pos):
        for idx, joint_name in self.jmsIdxToJoint.items():
            jms = JointMotorSettings(
                joint_pos,  # position_target
                0.09,  # position_gain
                0.0,  # velocity_target
                0.5,  # velocity_gain
                0.1,  # max_impulse
            )
            if joint_type in joint_name:
                self.robot_id.update_joint_motor(idx, jms)

    def _place_agent(self):
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
        # simulate dt seconds at 60Hz to the nearest fixed timestep
        observations = []
        for _ in range(steps):
            self._sim.step_physics(1.0 / self.ctrl_hz)
            vis = {}
            if render:
                vis = self._sim.get_sensor_observations()
            obs = self._read_sensors() #NOTE: It's important that this is run every time step physics is run so that the join velocity stays correct
            obs.update(vis)
            observations.append(obs)

        return observations
    
    def _set_jms_pos(self, joint_pos):
        for new_pos, jms in zip(joint_pos, self.jms_list):
            jms.position_target += new_pos
        
        for idx, jms in enumerate(self.jms_list):
            self.robot_id.update_joint_motor(idx, jms)

    def _read_sensors(self):
        obs = {}
        obs["joint_pos"] = np.array(self.robot_id.joint_positions)
        obs["joint_vel"] = (np.array(self.robot_id.joint_positions) - np.array(self.last_joint_pos)) / (1 / self.sim_hz)
        self.last_joint_pos = self.robot_id.joint_positions
        return obs

    def reset(self, render_episode=False):
        self.render_episode = render_episode
        self._reset_robot()

        # Let the robot fall to the ground
        obs = self._simulate(steps=60, render=render_episode)
        self.viz_buffer = obs
        self.episode_steps = 0

        observations = None

        return observations

    def step(self, *args, **kwargs):
        self.episode_steps += 1
        # self._set_jms_pos(args[0]["action_args"]) #NOTE: this seems a bit weird
        #NOTE: SIM_HZ/CTRL_HZ allows us to update motors at correct freq.
        assert int(self.sim_hz/self.ctrl_hz) == self.sim_hz/self.ctrl_hz, "sim_hz should be divis by ctrl_hz for simplicity"
        obs = self._simulate(steps=int(self.sim_hz/self.ctrl_hz), render=self.render_episode)[-1] 


        # print("joint pos: ", obs["joint_pos"])
        # print("joint vel: ", obs["joint_vel"])
        # print("-"*100)
        self.viz_buffer.append(obs)

        done = self.episode_steps >= 0
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
        reward = 0
        return obs, reward, done, info
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
