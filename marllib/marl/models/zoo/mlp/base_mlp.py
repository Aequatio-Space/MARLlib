# MIT License
import logging

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

import wandb
from ray.rllib.utils.torch_ops import FLOAT_MIN
from functools import reduce
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder

torch, nn = try_import_torch()


class BaseMLPMixin:

    def __init__(self):
        self.custom_config = None
        self.p_encoder = None
        self.p_branch = None
        self.vf_encoder = None
        self.vf_branch = None
        self.q_flag = False
        self.actors = None
        self.critics = None

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        observation = input_dict['obs']['obs']
        inf_mask = None
        if isinstance(observation, dict):
            flat_inputs = {k: v.float() for k, v in observation.items()}
        else:
            flat_inputs = observation.float()
        if self.custom_config["global_state_flag"] or self.custom_config["mask_flag"]:
            # Convert action_mask into a [0.0 || -inf]-type mask.
            if self.custom_config["mask_flag"]:
                action_mask = input_dict["obs"]["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        self.inputs = flat_inputs
        self._features = self.p_encoder(self.inputs)
        output = self.p_branch(self._features)

        if self.custom_config["mask_flag"]:
            output = output + inf_mask

        return output, state

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


class BaseMLP(TorchModelV2, nn.Module, BaseMLPMixin):
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
        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]
        self.activation = model_config.get("fcnet_activation")

        # encoder
        self.p_encoder = BaseEncoder(model_config, self.full_obs_space)
        self.vf_encoder = BaseEncoder(model_config, self.full_obs_space)

        self.p_branch = SlimFC(
            in_size=self.p_encoder.output_dim,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        # self.vf_encoder = nn.Sequential(*copy.deepcopy(layers))
        self.vf_branch = SlimFC(
            in_size=self.vf_encoder.output_dim,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        logging.debug(f"Encoder Configuration: {self.p_encoder}, {self.vf_encoder}")
        logging.debug(f"Branch Configuration: {self.p_branch}, {self.vf_branch}")
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_obs = None
        self.q_flag = False

        self.actors = [self.p_encoder, self.p_branch]
        self.critics = [self.vf_encoder, self.vf_branch]
        self.actor_initialized_parameters = self.actor_parameters()
        if wandb.run is not None:
            wandb.watch(models=tuple(self.actors), log='all')

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        return BaseMLPMixin.forward(self, input_dict, state, seq_lens)

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return BaseMLPMixin.value_function(self)


class CrowdSimMLP(TorchModelV2, nn.Module, BaseMLPMixin):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)
        self.embedding_dim = 32
        self.keys_fc = SlimFC(
            in_size=self.p_encoder.output_dim,
            out_size=self.embedding_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.query_fc = SlimFC(
            in_size=self.p_encoder.output_dim,
            out_size=self.embedding_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.sqrt_dim = torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32))
        self.reward_predict_fc = SlimFC(
            in_size=self.embedding_dim,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        observation = input_dict['obs']['obs']
        inf_mask = None
        if isinstance(observation, dict):
            flat_inputs = {k: v.float() for k, v in observation.items()}
        else:
            flat_inputs = observation.float()
        if self.custom_config["global_state_flag"] or self.custom_config["mask_flag"]:
            # Convert action_mask into a [0.0 || -inf]-type mask.
            if self.custom_config["mask_flag"]:
                action_mask = input_dict["obs"]["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        agents_features, emgergency_features = input_dict['obs']['obs']
        agents_embedding = self.keys_fc(agents_features)
        emgergency_embedding = self.query_fc(emgergency_features)
        # calculate the attention map for agents and emergency PoIs
        attention_map = torch.matmul(agents_embedding, emgergency_embedding.transpose(1, 2))
        # go through softmax
        attention_map = torch.softmax(attention_map / self.sqrt_dim, dim=1)
        attention_features = torch.matmul(attention_map, emgergency_features)
        predicted_reward = self.reward_predict_fc(attention_features)
        logits = self.p_branch(agents_features)

        if self.custom_config["mask_flag"]:
            logits = logits + inf_mask

        return logits, predicted_reward, state
