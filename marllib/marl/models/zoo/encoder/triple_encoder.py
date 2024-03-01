from collections import deque
from typing import List, Dict

import numpy as np
from gym import spaces
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()
from warp_drive.utils.constants import Constants


class TripleHeadEncoder(nn.Module):
    """
    Encoder for Special Mix input of crowdsim environment.
    """

    def __init__(self, custom_model_config, model_arch_args, obs_space):
        super(TripleHeadEncoder, self).__init__()
        self.custom_config = custom_model_config
        self.model_arch_args = model_arch_args
        self.obs_space = obs_space
        self.last_weight_matrix = None
        self.activation = self.custom_config.get("fcnet_activation")
        self.emergency_dim = self.custom_config['emergency_dim']
        # Two encoders, one for image and vector input, one for vector input.
        self.emergency_feature_dim = 2
        status_grid_space = dict(obs=spaces.Dict({
            'agents_state': spaces.Box(low=0, high=2, shape=(self.custom_config['status_dim'] +
                                                             self.emergency_feature_dim,), dtype=np.float32),
            'grid': spaces.Box(low=0, high=1, shape=eval(self.custom_config['grid_dim']), dtype=np.float32)
        }))
        emergency_obs_space = dict(obs=spaces.Box(low=0, high=2, shape=(self.emergency_dim,), dtype=np.float32))
        self.status_grid_encoder = BaseEncoder(self.model_arch_args['status_encoder'], status_grid_space)
        emergency_encoder_arch_args = self.model_arch_args['emergency_encoder']['custom_model_config'][
            'model_arch_args']
        self.emergency_encoder_arch = emergency_encoder_arch_args['core_arch']
        if self.emergency_encoder_arch == 'mlp':
            self.emergency_encoder = BaseEncoder(emergency_encoder_arch_args, emergency_obs_space)
        elif self.emergency_encoder_arch == 'attention':
            emergency_encoder_arch_args['emergency_queue_length'] = model_arch_args['emergency_queue_length']
            emergency_encoder_arch_args['agents_state_dim'] = (status_grid_space['obs']['agents_state'].shape[0] -
                                                               self.emergency_feature_dim)
            emergency_encoder_arch_args['num_heads'] = model_arch_args['num_heads']
            emergency_encoder_arch_args['attention_dim'] = model_arch_args['attention_dim'] * model_arch_args[
                'num_heads']
            self.emergency_encoder = CrowdSimAttention(emergency_encoder_arch_args)
        else:
            raise ValueError(f"Emergency encoder architecture {self.emergency_encoder_arch} not supported.")
        self.output_dim = self.emergency_encoder.output_dim + self.status_grid_encoder.output_dim

    def forward(self, inputs):
        status = inputs[Constants.VECTOR_STATE][..., :-self.emergency_dim + 2]
        emergency = inputs[Constants.VECTOR_STATE][..., -self.emergency_dim:]
        output = self.status_grid_encoder({Constants.VECTOR_STATE: status,
                                           Constants.IMAGE_STATE: inputs[Constants.IMAGE_STATE]})
        if self.emergency_encoder_arch == 'mlp':
            emergency_embedding = self.emergency_encoder(emergency)
        else:
            emergency_embedding, weights_matrix = self.emergency_encoder(
                {Constants.VECTOR_STATE: status[..., :-2], 'emergency': emergency})
            self.last_weight_matrix = weights_matrix
        status_embedding, grid_embedding = output[..., :self.status_grid_encoder.dims[0]], \
            output[..., self.status_grid_encoder.dims[0]:]
        # x = self.merge_branch(torch.cat((status_embedding, emergency, grid_embedding), dim=-1))
        return torch.cat((status_embedding, emergency_embedding, grid_embedding), dim=-1)


class CrowdSimAttention(nn.Module):
    """
    Attention Mechanism designed to select important emergency
    """

    def __init__(
            self,
            model_config,
    ):
        nn.Module.__init__(self)
        self.attention_dim = model_config['attention_dim']
        self.emergency_queue_length = model_config['emergency_queue_length']
        self.agents_state_dim = model_config['agents_state_dim']
        self.emergency_feature = 2
        self.num_heads = model_config['num_heads']
        self.keys_fc = SlimFC(
            in_size=self.emergency_feature,
            out_size=self.attention_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.values_fc = SlimFC(
            in_size=self.emergency_feature,
            out_size=self.attention_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.query_fc = SlimFC(
            in_size=self.agents_state_dim,
            out_size=self.attention_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.attention = nn.MultiheadAttention(self.attention_dim, self.num_heads, batch_first=True)
        self.output_dim = self.attention_dim

    def forward(self, input_dict: Dict[str, TensorType]) -> (TensorType, List[TensorType]):
        agents_state, emergency = tuple(input_dict.values())
        agents_embedding = self.query_fc(agents_state.unsqueeze(1))
        emergency = emergency.reshape(agents_state.shape[0], -1, 2)
        key_embedding = self.keys_fc(emergency)
        # change embedding dim.
        value_embedding = self.values_fc(emergency)
        attn_outputs, attn_output_weights = self.attention(agents_embedding, key_embedding, value_embedding)
        return attn_outputs.squeeze(1), attn_output_weights.squeeze(1)
