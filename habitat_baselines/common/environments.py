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
        # self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self.num_joints = self._core_env_config["num_joints"]
        # self._reward_measure_name = self._rl_config.REWARD_MEASURE
        # self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self._previous_action = None
        self.steps = 0
        self.action_space = ActionSpace(
            {
                "joint_targets": spaces.Box(
                    low=np.ones(self.num_joints) * np.deg2rad(-180),
                    high=np.ones(self.num_joints) * np.deg2rad(180),
                    dtype=np.float32,
                ),
            }
        )

        self.render_episode=False

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_path = os.path.join(dir_path, "../../../habitat-sim/data/")

        cfg = self._make_configuration()
        self.sim = habitat_sim.Simulator(cfg)
        self._place_agent(self.sim)
        urdf_files = {
            "aliengo": os.path.join(self.data_path, "URDF_demo_assets/aliengo/urdf/aliengo.urdf"),
        }
        # load a URDF file
        robot_file = urdf_files["aliengo"]
        ao_mgr = self.sim.get_articulated_object_manager()
        self.robot_id = ao_mgr.add_articulated_object_from_urdf(
            robot_file, fixed_base=False
        )
        self._place_robot()
        self.obs_buffer = []


        obj_template_mgr = self.sim.get_object_template_manager()
        rigid_obj_mgr = self.sim.get_rigid_object_manager()
        cube_handle = obj_template_mgr.get_template_handles("cubeSolid")[0]
        cube_template_cpy = obj_template_mgr.get_template_by_handle(cube_handle)
        cube_template_cpy.scale = np.array([5.0, 0.2, 5.0])
        obj_template_mgr.register_template(cube_template_cpy)
        ground_plane = rigid_obj_mgr.add_object_by_template_handle(cube_handle)
        ground_plane.translation = [0.0, -0.2, 0.0]
        ground_plane.motion_type = habitat_sim.physics.MotionType.STATIC
        # super().__init__(self._core_env_config, dataset)


        
    def _place_robot(self, angle_correction=-1.56, local_base_pos=None):
        base_transform = mn.Matrix4.rotation(
            mn.Rad(angle_correction), mn.Vector3(1.0, 0.0, 0.0)
        )
        base_transform.translation = mn.Vector3(0.0, 0.5, 0.0)
        self.robot_id.transformation = base_transform

    
    def _place_agent(self, sim):
        # place our agent in the scene
        agent_state = habitat_sim.AgentState()
        agent = sim.initialize_agent(0, agent_state)
        agent_state = agent.get_state()
        [print( mn.Vector3(*agent_state.rotation.imag).normalized()) for _ in range(10)]
        [print(agent_state.rotation.imag) for _ in range(10)]
        agent_rot = mn.Matrix4.from_(
            mn.Quaternion.rotation(
                mn.Rad(agent_state.rotation.real), mn.Vector3(*agent_state.rotation.imag).normalized()
            ).to_matrix(),
            mn.Vector3(0., 0., 0.)
        )
        [print(11) for _ in range(10)]
        pan_rot = mn.Matrix4.rotation(
            mn.Rad(np.deg2rad(-10)), mn.Vector3(0.0, 1.0, 0.0)
        )
        print("Agent rot, pan rot", agent_rot, pan_rot)
        agent_rot = agent_rot.__matmul__(pan_rot)
        agent_quat = mn.quaternion.from_matrix(agent_rot.rotation())
        agent_state.rotation = np.quaternion(agent_quat.scalar, *agent_quat.vector).normalized()
        agent_state.position = [-2.0, 0.5, 2.0]
        agent.set_state(agent_state)

        print("transofmration matrix: ", agent.scene_node.transformation_matrix())
        print("transofmration matrix: ", type(agent.scene_node.transformation_matrix()))
        # agent_state.position = [-0.15, -0.7, 1.0] 
        # agent_state.position = [-0.15, -1.6, 1.0]

        # mn_rot = mn.Matrix4.rotation_y(
        #     mn.Rad(np.deg2rad(45))
        # )
        # .__matmul__(
        #     mn.Matrix4.rotation_y(
        #         mn.Rad(np.deg2rad(0))
        #     )    
        # )
        # agent_state.rotation = np.quaternion(np.deg2rad(-45), 1, 0, 0)

        print("Rotation: ", np.rad2deg(agent_state.rotation.real), agent_state.rotation.imag)
        # agent_state.rotation.real = np.deg2rad(-90.0)
        # agent_state.rotation.imag = [1.0, 0.0, 0.0]
        # agent_state.rotation = agent_state.rotation.normalized()

        return agent.scene_node.transformation_matrix()


    def _make_configuration(self):
        # simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        # backend_cfg.scene_id = os.path.join(self.data_path, "scene_datasets/habitat-test-scenes/apartment_1.glb")
        backend_cfg.scene_id = ""
        backend_cfg.enable_physics = True

        # sensor configurations
        # Note: all sensors must have the same resolution
        # setup 2 rgb sensors for 1st and 3rd person views
        camera_resolution = [540, 720]
        sensors = {
            "rgba_camera_1stperson": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": camera_resolution,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
            },
            "depth_camera_1stperson": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": camera_resolution,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.orientation = sensor_params["orientation"]
            sensor_specs.append(sensor_spec)

        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        return habitat_sim.Configuration(backend_cfg, [agent_cfg])

    def _simulate(self, sim, steps=1, get_frames=False):
        # simulate dt seconds at 60Hz to the nearest fixed timestep
        # print("Simulating " + str(steps) + " world steps.")
        observations = []
        for _ in range(steps):
            sim.step_physics(1.0 / 60.0)
            if get_frames:
                observations.append(sim.get_sensor_observations())

        return observations

    def reset(self, render_episode=False):
        # self._previous_action = None
        # observations = super().reset()

        self.render_episode = render_episode
        self._place_robot()
        pose = self.robot_id.joint_positions
        print("0 POSE: ", pose)
        calfDofs = [2, 5, 8, 11]
        for dof in calfDofs:
            pose[dof] = np.deg2rad(-75) #second joint
            pose[dof - 1] = np.deg2rad(45) #first joint
        # pose[6] = -1
        self.robot_id.joint_positions = pose
        observations = self._simulate(self.sim, steps=60, get_frames=render_episode)
        self.obs_buffer += observations
        self.steps = 0
        return observations

    def step(self, *args, **kwargs):
        self.steps += 1
        self._previous_action = kwargs["action"]
        #TODO: set motor values here
        observations = self._simulate(self.sim, steps=1, get_frames=self.render_episode)
        self.obs_buffer += observations

        done = self.steps == 1000
        if done and self.render_episode:
            print("Rendering Vid")
            print("Obs: ", len(self.obs_buffer))
            vut.make_video(
                self.obs_buffer,
                "rgba_camera_1stperson",
                "color",
                "/nethome/skareer6/flash/Projects/InverseKinematics/AliengoGym/URDF_basics",
                open_vid=False,
            )

        info = {}
        reward = 0
        return observations, reward, done, info
        # return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        reward = self._env.get_metrics()[self._reward_measure_name]

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        # done = False
        # if self._env.episode_over or self._episode_success():
        #     done = True
        done = (self.steps % 250) == 0
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


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
