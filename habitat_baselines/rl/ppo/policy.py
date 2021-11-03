#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from habitat_baselines.rl.ppo import policy

import torch
from gym import spaces
from torch import nn as nn

from habitat.config import Config
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.utils.common import CategoricalNet, GaussianNet


class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions, policy_config=None):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        if policy_config is None:
            self.action_distribution_type = "categorical"
        else:
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        if self.action_distribution_type == "categorical":
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
        elif self.action_distribution_type == "gaussian":
            self.action_distribution = GaussianNet(
                self.net.output_size,
                self.dim_actions,
                policy_config.ACTION_DIST,
            )
        else:
            ValueError(
                f"Action distribution {self.action_distribution_type}"
                "not supported."
            )

        self.critic = CriticHead(self.net.output_size)
        self.critic_is_head = True

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        if self.critic_is_head:
            value = self.critic(features)
        else:
            value = self.critic(observations)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        if self.critic_is_head:
            return self.critic(features)

        return self.critic(observations)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        if self.critic_is_head:
            value = self.critic(features)
        else:
            value = self.critic(observations)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


@baseline_registry.register_policy
class LocomotionBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        mlp_hidden_sizes: list,
        hidden_size: int = 512,
        num_recurrent_layers: int = 3,
        policy_config=None,
        **kwargs,
    ):
        # Assume that each action is 1-dimensional
        num_actions = sum(
            [action.shape[0] for action in action_space.spaces.values()]
        )
        super().__init__(
            LocomotionBaselineNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                mlp_hidden_sizes=mlp_hidden_sizes,
                **kwargs,
            ),
            num_actions,
            policy_config=policy_config,
        )

        # Need to make a module that extracts data from a TensorDict
        extract_observation_values = nn.Module()
        extract_observation_values.forward = lambda x: torch.cat(
            list(x.values()),
            dim=1,
        )

        self.critic.fc = nn.Sequential(
            extract_observation_values,
            construct_mlp(self.net.non_visual_size, mlp_hidden_sizes),
            nn.Linear(mlp_hidden_sizes[-1], 1),
        )
        self.critic_is_head = False

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            mlp_hidden_sizes=config.RL.PPO.mlp_hidden_sizes,
            hidden_size=config.RL.PPO.hidden_size,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            policy_config=config.RL.POLICY,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


def construct_mlp(input_size, mlp_hidden_sizes):
    layers = []
    previous_hidden_units = input_size

    # Stack of Linear and ReLUs
    for hidden_units in mlp_hidden_sizes:
        layers.append(nn.Linear(previous_hidden_units, hidden_units))
        layers.append(nn.ReLU())
        previous_hidden_units = hidden_units

    return nn.Sequential(*layers)


class LocomotionBaselineNet(Net):
    r"""Network which passes the input image through CNN if provided and concatenates
    goal vector with CNN's output and passes that through RNN.

    NOTE: for simplicity I'm assuming that observation_space is not a nested dict.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
        num_recurrent_layers: int,
        mlp_hidden_sizes: list,
    ):
        super().__init__()

        # Assuming every observation type/key is 1D, and we use ALL of them
        self.non_visual_size = sum(
            [v.shape[0] for v in observation_space.spaces.values()]
        )

        self._hidden_size = hidden_size

        # Can't delete this... RolloutStorage uses this to know hidden size.
        self.state_encoder = build_rnn_state_encoder(
            self.non_visual_size,
            hidden_size=self._hidden_size,
            num_layers=num_recurrent_layers,
        )

        self._mlp_output_size = mlp_hidden_sizes[-1]
        self.mlp = construct_mlp(self.non_visual_size, mlp_hidden_sizes)

        self.train()

    @property
    def output_size(self):
        return self._mlp_output_size

    @property
    def is_blind(self):
        return True  # NOTE: need to change if we add back CNN

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = list(observations.values())
        x_out = torch.cat(x, dim=1)
        x_out = self.mlp(x_out)

        # Dummy hidden state
        rnn_hidden_states = torch.zeros(
            x_out.shape[0],  # number of environments
            self.state_encoder.num_recurrent_layers,
            self._hidden_size,
            device=x_out.device,
        )

        return x_out, rnn_hidden_states


@baseline_registry.register_policy
class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        policy_config=None,
        **kwargs,
    ):
        super().__init__(
            PointNavBaselineNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                **kwargs,
            ),
            action_space.n,
            policy_config=policy_config,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            policy_config=config.RL.POLICY,
        )


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
    ):
        super().__init__()

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            self._n_input_goal = observation_space.spaces[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ].shape[0]
        elif PointGoalSensor.cls_uuid in observation_space.spaces:
            self._n_input_goal = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
        elif ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = SimpleCNN(
                goal_observation_space, hidden_size
            )
            self._n_input_goal = hidden_size

        self._hidden_size = hidden_size

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        """ LOCOMOTION HACK """
        self.goal_encoder = nn.Sequential(
            nn.Linear(12, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self._n_input_goal = 256

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        # if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
        #     target_encoding = observations[
        #         IntegratedPointGoalGPSAndCompassSensor.cls_uuid
        #     ]
        #
        # elif PointGoalSensor.cls_uuid in observations:
        #     target_encoding = observations[PointGoalSensor.cls_uuid]
        # elif ImageGoalSensor.cls_uuid in observations:
        #     image_goal = observations[ImageGoalSensor.cls_uuid]
        #     target_encoding = self.goal_visual_encoder({"rgb": image_goal})
        #
        # x = [target_encoding]
        #
        # if not self.is_blind:
        #     perception_embed = self.visual_encoder(observations)
        #     x = [perception_embed] + x

        """LOCOMOTION HACK"""
        x = [observations["joint_pos"]]

        x_out = torch.cat(x, dim=1)
        x_out = self.goal_encoder(x_out)
        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks
        )

        return x_out, rnn_hidden_states