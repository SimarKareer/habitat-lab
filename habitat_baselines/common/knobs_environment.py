import habitat
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.spaces import ActionSpace
from gym import spaces
import numpy as np
import torch
from habitat.utils.geometry_utils import wrap_heading


@baseline_registry.register_env(name="KnobsEnv")
class KnobsEnv(habitat.RLEnv):

    """Agent has to move knobs to goal positions!"""

    def __init__(self, config=None, *args, **kwargs):
        self.env_config = config.TASK_CONFIG.ENVIRONMENT
        self.num_knobs = self.env_config.NUM_KNOBS
        self.success_thresh = np.deg2rad(self.env_config.SUCCESS_THRESH)
        self._max_episode_steps = self.env_config.MAX_STEPS

        # Create action space
        self.max_movement = np.deg2rad(self.env_config.MAX_MOVEMENT)
        # Agent has an action for each knob, (how much each knob is changed by)
        self.action_space = ActionSpace(
            {
                "knob_deltas": spaces.Box(
                    low=-self.max_movement,
                    high=self.max_movement,
                    shape=(self.num_knobs,),
                    dtype=np.float32,
                )
            }
        )

        # Create observation space
        # Agent is given the current angle difference of each knob (-180 - 180)
        self.observation_space = spaces.Dict(
            {
                "knob_errors": spaces.Box(
                    low=-np.pi,
                    high=np.pi,
                    shape=(self.num_knobs,),
                    dtype=np.float32,
                ),
            }
        )

        self.current_state = None
        self.goal_state = None
        self.num_steps = 0
        self.cumul_reward = 0

        self.number_of_episodes = int(1e24)  # needed b/c habitat wants it

    def close(self):  # needed b/c habitat wants it
        pass

    def seed(self, seed):  # needed b/c habitat wants it
        torch.manual_seed(seed)
        np.random.seed(seed)

    def reset(self, goal_state=None):
        # Randomize goal state if not specified
        if goal_state is None:
            goal_state = self._get_random_knobs()

        self.goal_state = goal_state
        self.current_state = self._get_random_knobs()
        self.num_steps = 0
        observations = self.get_observations()
        self.cumul_reward = 0

        return observations

    def get_observations(self):
        return {
            "knob_errors": np.array(
                [
                    self._get_heading_error(c, g)
                    for c, g in zip(self.current_state, self.goal_state)
                ]
            )
        }

    def _get_random_knobs(self):
        # (0, 1) -> (0, 2) -> (-1, 1) -> (-np.pi, np.pi)
        return (np.random.rand(self.num_knobs) * 2 - 1) * np.pi

    def step(self, action, action_args, *args, **kwargs):

        # Clip actions and scale
        action = (
            np.clip(action_args["knob_deltas"], -1.0, 1.0) * self.max_movement
        )
        # action = np.ones(len(action), dtype=np.float32) * self.max_movement

        # Update current state
        self.current_state = np.array(
            [
                self._validate_heading(c + a)
                for c, a in zip(self.current_state, action)
            ]
        )

        # Return observations (error for each knob)
        observations = self.get_observations()

        # Penalize MSE between current and goal states
        reward_terms = -(observations["knob_errors"] ** 2) / self.num_knobs
        reward = sum(reward_terms)

        # Check termination conditions
        success = all(
            [abs(i) < self.success_thresh for i in observations["knob_errors"]]
        )

        self.num_steps += 1
        done = success or self.num_steps == self._max_episode_steps

        self.cumul_reward += reward
        info = {
            "reward": reward,
            "success": success,
            "failed": self.num_steps == self._max_episode_steps,
            "cumul_reward": self.cumul_reward,
            "reward_terms": reward_terms,
            "episode": {"r": reward},
        }

        return observations, reward, done, info

    def _validate_heading(self, heading):
        if heading > np.pi:
            heading -= np.pi * 2
        elif heading < -np.pi:
            heading += np.pi * 2

        return heading

    def _get_heading_error(self, source, target):
        return self._validate_heading(target - source)
