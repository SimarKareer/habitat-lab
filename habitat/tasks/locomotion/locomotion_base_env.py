import os
from collections import defaultdict

import numpy as np
import quaternion as qt
import torch
from gym import spaces

import habitat
import habitat_sim
import habitat_sim.utils.viz_utils as vut
from habitat import Config
from habitat.core.spaces import ActionSpace
from habitat.tasks.locomotion.aliengo import AlienGo
from habitat_baselines.common.baseline_registry import baseline_registry


class ActionType:
    RELATIVE = "relative"
    ABSOLUTE = "absolute"


@baseline_registry.register_env(name="LocomotionRLEnv")
class LocomotionRLEnv(habitat.RLEnv):
    def __init__(
        self, config: Config, render=False, num_robots=1, *args, **kwargs
    ):
        self.config = config
        self.sim_config = config.TASK_CONFIG.SIMULATOR
        self.task_config = config.TASK_CONFIG
        self.robot_config = config.TASK_CONFIG.ROBOT
        self.num_robots = num_robots

        # Create action space
        self.num_joints = self.task_config.TASK.ACTION.NUM_JOINTS
        self.max_rad_delta = np.deg2rad(
            self.task_config.TASK.ACTION.MAX_DEGREE_DELTA
        )

        self.action_space = ActionSpace(
            {
                "joint_deltas": spaces.Box(
                    low=-1, high=1, shape=(self.num_joints,), dtype=np.float32
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
                    low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32
                ),
                "feet_contact": spaces.Box(
                    low=0, high=1, shape=(4,), dtype=np.float32
                ),
            }
        )

        # Create simulator
        self.floor_ids = []
        self._sim = self._create_sim()

        # Place agent (for now, this is just a camera)
        self._place_agent()

        # Debug
        self.video_dir = config.VIDEO_DIR

        # Load the robot into the sim
        self.ao_mgr = self._sim.get_articulated_object_manager()
        self._load_robots()

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
        self.render_episode = False
        self.sim_hz = self.sim_config.SIM_HZ
        self.number_of_episodes = int(1e24)  # needed b/c habitat wants it
        self.accumulated_reward_info = defaultdict(float)
        self.action_type = config.TASK_CONFIG.TASK.ACTION.TYPE
        self.jitter_threshold = config.TASK_CONFIG.TASK.JITTER_THRESHOLD

    def reset(self):
        self._task_reset()

        # Let robot settle on the ground
        self._sim.step_physics(self.settle_time)

        self.viz_buffer = []
        self.num_steps = 0
        self.accumulated_reward_info = defaultdict(float)

        return self._get_observations()

    def step(self, action, action_args, step_render=False, *args, **kwargs):
        """Updates robot with given actions and calls physics sim step"""
        if self.task_config.DEBUG.BASELINE_POLICY:
            deltas = self._baseline_policy()
            self.robot.add_jms_pos(deltas)
        else:
            # Update current state
            if self.action_type == ActionType.RELATIVE:
                action_scale = self.max_rad_delta
                offset = self.robot.joint_positions
            elif self.action_type == ActionType.ABSOLUTE:
                action_scale = self.robot.joint_limits_stand
                offset = self.robot.standing_pose
            else:
                raise NotImplementedError(f"{self.action_type} unknown!")
            deltas = action_args["joint_deltas"]
            scaled_actions = np.clip(deltas, -1.0, 1.0) * action_scale
            target_pose = scaled_actions + offset

            # Remove jitter by maintaining current joint for too-close targets
            jitter_inds = (
                abs(target_pose - self.robot.joint_positions)
                < self.jitter_threshold
            )
            target_pose[jitter_inds] = self.robot.joint_positions[jitter_inds]

            self.robot.set_pose_jms(target_pose, kinematic_snap=False)

        self._sim.step_physics(1.0 / self.sim_hz)

        # Return observations
        observations = self._get_observations()

        # Get reward
        reward_terms = self._get_reward_terms(observations)
        reward = sum(reward_terms)

        # Text on Screen
        if self.render:
            self.viz_buffer.append(self._sim.get_sensor_observations())
            self.viz_buffer[-1]["rgba_camera"] = self.image_text(
                self.viz_buffer[-1]["rgba_camera"]
            )

        # Check termination conditions
        self.num_steps += 1
        done = self.num_steps == self._max_episode_steps or self.should_end()

        # self.render needs to be on, but because of this self.should_end
        # is causing it to be done every step
        if done and self.render or step_render:
            vut.make_video(
                self.viz_buffer,
                "rgba_camera",
                "color",
                os.path.join(
                    self.video_dir, f"vid{np.random.randint(0, 1e6)}.mp4"
                ),
                open_vid=False,
                fps=self.sim_hz,
            )

        # Populate info for tensorboard
        info = {
            "success": 1.0 if self._get_success() else 0.0,
            "height": self.robot.height,
            "reward_terms": reward_terms,
        }
        info.update(self._get_extra_info())

        # Add info about how much of each reward component has been accumulated
        info.update(self.accumulated_reward_info)

        return observations, reward, done, info

    def _place_agent(self):
        """Places our camera agent in a spot it can see the robot"""
        # place our agent in the scene
        agent_state = habitat_sim.AgentState()
        agent_state.position = [-1.1, 0.5, 1.1]
        agent_state.rotation *= qt.from_euler_angles(0.0, np.deg2rad(-50), 0.0)
        agent = self._sim.initialize_agent(0, agent_state)

        return agent.scene_node.transformation_matrix()

    def _create_sim(self):
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
            rgb_camera.resolution = [540, 720]
            # These are relative to the agent, not to global origin
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
        cube_template_cpy.scale = np.array([10.0, 0.2, 10.0])
        cube_template_cpy.friction_coefficient = 0.5
        obj_template_mgr.register_template(cube_template_cpy)
        rigid_obj_mgr = sim.get_rigid_object_manager()
        floor_height = -0.2
        for idx in range(self.num_robots):
            self.floor_ids.append(
                rigid_obj_mgr.add_object_by_template_handle(cube_handle)
            )
            self.floor_ids[idx].translation = [0.0, floor_height, 0.0]
            self.floor_ids[
                idx
            ].motion_type = habitat_sim.physics.MotionType.KINEMATIC
            if idx < self.num_robots - 1:
                floor_height += self.task_config.SIMULATOR.FLOOR_SEPARATION

        return sim

    def _load_robots(self):
        self.fixed_base = self.task_config.DEBUG.FIXED_BASE
        robot_id = self.ao_mgr.add_articulated_object_from_urdf(
            self.task_config.ROBOT.ROBOT_URDF, fixed_base=self.fixed_base
        )
        self.robot = AlienGo(
            robot_id, self._sim, self.fixed_base, self.task_config.ROBOT
        )

    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _get_observations(self):
        return {
            "joint_pos": self.robot.joint_positions,
            "joint_vel": self.robot.joint_velocities,
            "euler_rot": self.robot.get_rp(),
            "feet_contact": self.robot.get_feet_contacts(),
        }

    def image_text(self, img):
        return img

    def should_end(self):
        return False

    def _task_reset(self):
        f"""Task specific robot position reset"""
        raise NotImplementedError

    def _baseline_policy(self):
        raise NotImplementedError

    def _get_success(self):
        return False

    def _get_extra_info(self):
        return {}

    def _get_reward_terms(self, observations) -> np.array:
        raise NotImplementedError

    def close(self):
        pass
