import magnum as mn
import numpy as np

from habitat.tasks.locomotion.aliengo_vectorized import AlienGoVectorized
from habitat.tasks.locomotion.locomotion_base_env import (
    ActionType,
    LocomotionRLEnv,
)
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="LocomotionRLEnv")
class LocomotionVectorRLEnv(LocomotionRLEnv):
    is_vector_env = True

    def __init__(self, config, *args, **kwargs):
        num_robots = config.NUM_ENVIRONMENTS
        self.robot_ids = []
        self.robot = None
        super().__init__(config, num_robots=num_robots, *args, **kwargs)
        self.num_steps = np.zeros(self.num_robots)

    def _load_robots(self):
        """ "Load in all robots with different height offsets"""
        self.fixed_base = self.task_config.DEBUG.FIXED_BASE
        height_offset = mn.Vector3(
            0.0, self.robot_config.ROBOT_SPAWN_HEIGHT, 0.0
        )
        reset_positions = []
        for idx in range(self.num_robots):
            self.robot_ids.append(
                self.ao_mgr.add_articulated_object_from_urdf(
                    self.task_config.ROBOT.ROBOT_URDF,
                    fixed_base=self.fixed_base,
                )
            )
            reset_positions.append(
                self.floor_ids[idx].translation + height_offset
            )
        self.robot = AlienGoVectorized(
            self.robot_ids,
            self._sim,
            self.fixed_base,
            self.robot_config,
            reset_positions,
        )

    def step(self, action, action_args, step_render=False, *args, **kwargs):
        """Un-flatten actions"""
        observations, reward, done, info = super().step(
            action, action_args, step_render=False, *args, **kwargs
        )
        observations = self.reset_done_envs(done, observations)

        return observations, reward, done, info

    def get_done_info(self, reward_terms, step_render=False):
        done = np.zeros(self.num_robots)
        done[self.num_steps == self._max_episode_steps] = 1.0
        # Reset num steps
        self.num_steps[self.num_steps == self._max_episode_steps] = 0.0
        info = {"reward_terms": reward_terms}
        return done, info

    def step_physics(self, seconds):
        """Refresh the cache of matrices containing robot proprioception"""
        super().step_physics(seconds)
        self.robot.clear_cache()

    def reset_done_envs(self, done, observations):
        raise NotImplementedError
