# MIT License
import logging

import numpy as np
from gym import spaces
# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ray.rllib.utils.torch_ops import FLOAT_MIN
from functools import reduce
import copy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder, ResNetCNNEncoder

torch, nn = try_import_torch()


class CrowdSimNet(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # decide the model arch
        self.inputs = None
        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]
        self.activation = model_config.get("fcnet_activation")

        # Three encoders, two for vector input, one for image input
        status_obs_space = spaces.Box(low=0, high=1, shape=(self.custom_config['status_dim']), dtype=np.float32)
        emergency_obs_space = spaces.Box(low=0, high=2, shape=(self.custom_config['emergency_dim']), dtype=np.float32)
        self.status_encoder = BaseEncoder(status_obs_space, self.custom_config['status_encoder'])
        self.emergency_encoder = BaseEncoder(emergency_obs_space, self.custom_config['emergency_encoder'])
        self.grid_encoder = ResNetCNNEncoder(self.full_obs_space["grid"], self.custom_config['grid_encoder'])

        # Merge the three branches
        full_feature_output = self.custom_config["hidden_state_size"][0] * 3
        self.merge_branch = SlimFC(
            in_size=full_feature_output,
            out_size=self.custom_config["hidden_state_size"][0],
            initializer=normc_initializer(1.0),
            activation_fn=self.activation,
        )

        # Value Function
        self.value_branch = SlimFC(
            in_size=full_feature_output,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None,
        )

        # Policy Network
        self.actor_branch = SlimFC(
            in_size=full_feature_output,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        state, grid = tuple(input_dict.values())
        # slice the state to two parts, the last 4 dim is the emergency coordinate
        emergency = state[:, -self.custom_config['emergency_dim']:]
        status = state[:, :-self.custom_config['emergency_dim']]
        # encode the three branches
        status = self.status_encoder(status)
        emergency = self.emergency_encoder(emergency)
        grid = self.grid_encoder(grid)
        # merge the three branches
        x = torch.cat((status, emergency, grid), dim=1)
        x = self.merge_branch(x)
        # through actor branch
        logits = self.actor_branch(x)
        return logits

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        B = self._features.shape[0]
        x = self.vf_encoder(self.inputs)

        if self.q_flag:
            return torch.reshape(self.vf_branch(x), [B, -1])
        else:
            return torch.reshape(self.vf_branch(x), [-1])

    def actor_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    def critic_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.critics))
