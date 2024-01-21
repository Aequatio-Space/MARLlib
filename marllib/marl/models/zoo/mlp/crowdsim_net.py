# MIT License

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

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List
from marllib.marl.models.zoo.mlp.base_mlp import BaseMLPMixin

torch, nn = try_import_torch()


class CrowdSimNet(TorchModelV2, nn.Module, BaseMLPMixin):
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
        BaseMLPMixin.__init__(self)

        # decide the model arch
        self.inputs = None
        self._features = None
        self.q_flag = False
        self.custom_config = model_config["custom_model_config"]
        self.model_arch_args = self.custom_config["model_arch_args"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]
        self.activation = model_config.get("fcnet_activation")
        self.custom_config['fcnet_activaiton'] = self.activation
        # Value and Policy Encoder
        self.vf_encoder = TripleHeadEncoder(self.custom_config, self.model_arch_args, self.full_obs_space)
        self.p_encoder = TripleHeadEncoder(self.custom_config, self.model_arch_args, self.full_obs_space)
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
        return BaseMLPMixin.forward(self, input_dict, state, seq_lens)

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return BaseMLPMixin.value_function(self)
