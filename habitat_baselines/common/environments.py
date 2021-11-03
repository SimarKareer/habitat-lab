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
import habitat_sim.utils.viz_utils as vut
import numpy as np
import os

import habitat
import habitat_sim
from habitat import Config, Dataset
from habitat.core.spaces import ActionSpace
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
import quaternion as qt
from habitat_sim.physics import JointMotorSettings
from habitat.utils.geometry_utils import wrap_heading
import torch
from habitat_baselines.common.aliengo import AlienGo


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
    def __init__(self, config: Config, dataset: Optional[Dataset] = None, *args, **kwargs):
        # Config binding
        self.config = config
        self.sim_config = config.TASK_CONFIG.SIMULATOR
        self.task_config = config.TASK_CONFIG.TASK
        self.debug_config = config.TASK_CONFIG.DEBUG

        # Debug flags
        self.render_freq = self.debug_config.VIDEO_FREQ
        if self.render_freq == -1:
            self.render_freq = None
        self.fixed_base = self.debug_config.FIXED_BASE
        self.video_dir = config.VIDEO_DIR

        # Set up actions
        self.num_joints = config.TASK_CONFIG.TASK.ACTION.NUM_JOINTS
        self.max_rad_delta = np.deg2rad(config.TASK_CONFIG.TASK.ACTION.MAX_DEGREE_DELTA)
        self.action_space = ActionSpace({
            "joint_deltas": spaces.Box(
                low=np.ones(self.num_joints) * -self.max_rad_delta,
                high=np.ones(self.num_joints) * self.max_rad_delta,
                dtype=np.float32,
            )}
        )
        self.observation_space = spaces.Dict(
            {
                "joint_pos": spaces.Box(low=np.ones(self.num_joints) * -np.pi, high=np.ones(self.num_joints) * np.pi, dtype=np.float32),
                # "joint_vel": spaces.Box(low=np.ones(self.num_joints) * -1000, high=np.ones(self.num_joints) * 1000, dtype=np.float32),
                # "euler_rot": spaces.Box(low=np.ones(3) * -np.pi, high=np.ones(3) * np.pi, dtype=np.float32)
            }
        )

        # Create sim
        cfg = self._make_configuration()
        self._sim = habitat_sim.Simulator(cfg)

        # Place agent
        self._place_agent()

        # Load the robot into the sim
        ao_mgr = self._sim.get_articulated_object_manager()
        self.robot_id = ao_mgr.add_articulated_object_from_urdf(
            self.sim_config.ROBOT_URDF, fixed_base=self.fixed_base
        )
        self.robot = AlienGo(
            self.robot_id, self._sim, self.fixed_base, self.task_config
        )
        self._load_floor()

        # RL init
        self.target_joint_positions = np.array([0, 0.432, -0.77, 0, 0.432, -0.77, 0, 0.432, -0.77, 0, 0.432, -0.77])[:3]
        self.number_of_episodes = 1 if self.render_freq == 1 else int(1e6) #this signifies eval mode

        # joint position limits
        # self.joint_limits_lower = np.array([-0.1, -np.pi/3, -5/6*np.pi] * 4)[:3]
        # self.joint_limits_upper = np.array([0.1, np.pi/2.1, -np.pi/4] * 4)[:3]

        # Initialize attributes
        self.viz_buffer = []
        self.num_resets = 0
        self.episode_steps = 0
        self.render_episode = False
        self.last_joint_pos = self.robot.robot_id.joint_positions
        self.sim_hz = self.sim_config.SIM_HZ
        self.ctrl_hz = self.sim_config.CTRL_HZ

        # self._sim.set_gravity([0., 0., 0.])
        # super().__init__(self._core_env_config, dataset)

    def _load_floor(self):
        r"""load floor into the scene"""
        obj_template_mgr = self._sim.get_object_template_manager()
        cube_handle = obj_template_mgr.get_template_handles("cubeSolid")[0]
        cube_template_cpy = obj_template_mgr.get_template_by_handle(cube_handle)
        cube_template_cpy.scale = np.array([5.0, 0.2, 5.0])
        obj_template_mgr.register_template(cube_template_cpy)
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        ground_plane = rigid_obj_mgr.add_object_by_template_handle(cube_handle)
        ground_plane.translation = [0.0, -0.2, 0.0]
        ground_plane.motion_type = habitat_sim.physics.MotionType.STATIC

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
        backend_cfg.gpu_device_id = self.config.SIMULATOR_GPU_ID 
        backend_cfg.enable_physics = True

        # sensor configurations
        rgb_camera = habitat_sim.CameraSensorSpec()
        rgb_camera.uuid = "rgba_camera"
        rgb_camera.sensor_type = habitat_sim.SensorType.COLOR

        # Make camera res super low if we're not rendering
        if self.render_freq == None:
            rgb_camera.resolution = [1, 1]
            # rgb_camera.resolution = [540, 720]#HACK
        else:
            rgb_camera.resolution = [540, 720]

        rgb_camera.position = [0.0, 0.0, 0.0]
        rgb_camera.orientation = [0.0, 0.0, 0.0]

        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_camera]

        return habitat_sim.Configuration(backend_cfg, [agent_cfg])

    def close(self):
        pass

    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)


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
                self.viz_buffer.append(vis)
            obs = self._read_sensors() #NOTE: It's important that this is run every time step physics is run so that the join velocity stays correct
            # obs.update(vis)
            observations.append(obs)

        return observations
    
    def _read_sensors(self):
        """ Returns sensor observation for joint positions and velocities
        """
        obs = {}
        obs["joint_pos"] = np.array(self.robot_id.joint_positions)[:3]
        # obs["joint_vel"] = (wrap_heading(np.array(self.robot_id.joint_positions) - np.array(self.last_joint_pos)) / (1 / self.sim_hz))[:3]
        # obs["euler_rot"] = self.robot.get_robot_rpy()
        self.last_joint_pos = self.robot_id.joint_positions[:3]
        return obs
    
    def _step_reward(self, step_obs):
        joint_pos = np.array(step_obs["joint_pos"])
        target_joint_pos = np.array(self.target_joint_positions)
        # mse = (joint_pos - target_joint_pos)**2
        # mse = 0.5 * (wrap_heading(joint_pos - target_joint_pos) ** 2).mean() # MSE loss
        mseexp = np.exp(-5.0 * (wrap_heading(joint_pos - target_joint_pos) ** 2).sum()) # sehoon's loss
        # print("MSE: ", mse)

        # reward = -mse #negative if using MSE.  pos if using exp mse
        reward = mseexp
        # if (np.abs(step_obs["euler_rot"]) > np.deg2rad(60)).any():
            # assert(not self.fixed_base)
            # print("TIP PENALTY INCURRED")
            # reward -= self.task_config.REWARD.TIP_PENALTY
            # reward -= 1
        
        return reward

    def reset(self):
        """ Resets episode by resetting robot, visualization buffer, and allowing robot to fall to ground

            Args:
                render_epsiode: whether or not to render video for this episode
            
            Returns:
                None
        """
        self.render_episode = self.render_freq != None and (self.num_resets % self.render_freq == 0)
        self.robot.reset()

        self.viz_buffer = []
        # Let the robot fall to the ground
        obs = self._simulate(steps=80, render=self.render_episode)
        # print("INIT REWARD: ", self._step_reward(obs[0]))
        self.episode_steps = 0
        self.num_resets += 1
        if self.num_resets % 100 == 0:
            print("EPISODE NUM: ", self.num_resets)

        step_obs = obs[-1]
        # step_obs.pop("rgba_camera", None)
        # print("STEP OBS: ", step_obs)

        return step_obs

    def step(self, action, action_args, *args, **kwargs):
        """ Updates robot with given actions and calls physics sim step
        """
        # print("ACTIONS: ", action["action_args"])
        # deltas = action["action_args"]["joint_deltas"]
        # print("ACTION: ", action)
        deltas = action_args["joint_deltas"]
        assert(len(deltas) == self.num_joints)
        deltas = np.clip(deltas, -1, 1) * self.max_rad_delta
        self.episode_steps += 1
        # print("ARGS: ", args) #TODO: RM
        # print("KWARGS: ", kwargs) #TODO: RM
        self.robot.add_jms_pos(deltas) #NOTE: this seems a bit weird
        #NOTE: SIM_HZ/CTRL_HZ allows us to update motors at correct freq.
        assert int(self.sim_hz/self.ctrl_hz) == self.sim_hz/self.ctrl_hz, "sim_hz should be divis by ctrl_hz for simplicity"
        obs = self._simulate(steps=int(self.sim_hz/self.ctrl_hz), render=self.render_episode)


        # print("joint pos: ", obs["joint_pos"])
        # print("joint vel: ", obs["joint_vel"])
        # print("-"*100)
        # self.viz_buffer += obs #+= if obs is array, append else
        step_obs = obs[-1]
        # step_obs.pop("rgba_camera", None)
        # print("Robot RYP: ", np.rad2deg(step_obs["euler_rot"]))
        # print("STEPS AND RENDER: ", self.episode_steps, self.render_freq, self.config.RL.DDPPO.pretrained)

        done = self.episode_steps >= 500
        # self.render_episode = True #HACK: hack to get it to render
        # self.video_dir = "/coc/testnvme/skareer6/Projects/InverseKinematics/stand_exp/videos5e-4" #HACK
        if done and self.render_episode: 
            print("EP done!!!!: ", self.viz_buffer[0])
            vut.make_video(
                self.viz_buffer,
                "rgba_camera",
                "color",
                os.path.join(self.video_dir, "vid{num_resets}-{rand}.mp4".format(num_resets=self.num_resets, rand=np.random.randint(0, 1e6))),
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
