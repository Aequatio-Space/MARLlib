# MIT License
import logging

import numpy as np
from gym import spaces
# Copyright (c) 2023 Replicable-MARL
from ray.rllib.utils.torch_ops import FLOAT_MIN
from warp_drive.utils.constants import Constants
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

from functools import reduce
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
        self._features = None
        self.q_flag = False
        self.custom_config = model_config["custom_model_config"]
        self.model_arch_args = self.custom_config["model_arch_args"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]
        self.activation = model_config.get("fcnet_activation")

        # Two encoders, one for image and vector input, one for vector input.
        status_grid_space = dict(obs=spaces.Dict({
            'agents_state': spaces.Box(low=0, high=2, shape=eval(self.custom_config['status_dim']), dtype=np.float32),
            'grid': spaces.Box(low=0, high=1, shape=eval(self.custom_config['grid_dim']), dtype=np.float32)
        }))
        emergency_obs_space = dict(obs=spaces.Box(low=0, high=2, shape=eval(self.custom_config['emergency_dim']),
                                                  dtype=np.float32))
        self.status_grid_encoder = BaseEncoder(self.model_arch_args['status_encoder'], status_grid_space)
        self.emergency_encoder = BaseEncoder(self.model_arch_args['emergency_encoder'], emergency_obs_space)
        self.emergency_dim = eval(self.custom_config['emergency_dim'])[0]
        # Merge the three branches
        full_feature_output = self.emergency_encoder.output_dim + self.status_grid_encoder.output_dim
        self.merge_branch = SlimFC(
            in_size=full_feature_output,
            out_size=self.custom_config["hidden_state_size"],
            initializer=normc_initializer(1.0),
            activation_fn=self.activation,
        )

        # Value Function
        self.vf_branch = SlimFC(
            in_size=self.custom_config["hidden_state_size"],
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None,
        )

        # Policy Network
        self.p_branch = SlimFC(
            in_size=self.custom_config["hidden_state_size"],
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        observation = input_dict['obs']['obs']
        inf_mask = None
        if isinstance(observation, dict):
            flat_inputs = {k: v.float() for k, v in observation.items()}
        else:
            flat_inputs = observation.float()
        self.inputs = flat_inputs
        if self.custom_config["global_state_flag"] or self.custom_config["mask_flag"]:
            # TODO: Mask untested.
            # Convert action_mask into a [0.0 || -inf]-type mask.
            if self.custom_config["mask_flag"]:
                action_mask = input_dict["obs"]["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        # TODO: does not support 1D state
        # slice the state to two parts, the last 4 dim is the emergency coordinate
        status = self.inputs[Constants.VECTOR_STATE][..., :-self.emergency_dim]
        emergency = self.inputs[Constants.VECTOR_STATE][..., -self.emergency_dim:]
        # encode the three branches
        status = self.status_grid_encoder({Constants.VECTOR_STATE: status,
                                           Constants.IMAGE_STATE: flat_inputs[Constants.IMAGE_STATE]})
        emergency = self.emergency_encoder(emergency)
        # merge the three branches
        self._features = torch.cat((status, emergency), dim=1)
        x = self.merge_branch(self._features)
        # through actor branch
        logits = self.p_branch(x)
        if self.custom_config["mask_flag"]:
            logits = logits + inf_mask
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        B = self._features.shape[0]
        # x = self.vf_encoder(self.inputs)
        x = self.merge_branch(self._features)
        # WARNING: value function now shares encoder with policy, does not know the implication.
        if self.q_flag:
            return torch.reshape(self.vf_branch(x), [B, -1])
        else:
            return torch.reshape(self.vf_branch(x), [-1])

    def actor_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    def critic_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.critics))
