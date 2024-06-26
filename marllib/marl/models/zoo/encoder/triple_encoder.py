import logging

import numpy as np
from gym import spaces
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder
from ray.rllib.utils import try_import_torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from .attention_encoder import CrowdSimAttention

torch, nn = try_import_torch()
from warp_drive.utils.constants import Constants
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: torch.Tensor, temperature=1):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class TripleHeadEncoder(nn.Module):
    """
    Encoder for Special Mix input of crowdsim environment.
    """

    def __init__(self, custom_model_config, model_arch_args, obs_space):
        super(TripleHeadEncoder, self).__init__()
        self.custom_config = custom_model_config
        self.model_arch_args = model_arch_args
        emergency_encoder_arch_args = self.model_arch_args['emergency_encoder']['custom_model_config'][
            'model_arch_args']
        self.obs_space = obs_space
        self.last_weight_matrix = self.last_executor_obs = self.last_selection = None
        self.activation = self.custom_config.get("fcnet_activation")
        self.emergency_dim = self.custom_config['emergency_dim']
        self.emergency_queue_length = model_arch_args['emergency_queue_length']
        self.emergency_encoder_arch = emergency_encoder_arch_args['core_arch']
        # mlp, mlp_residual, attention, attention_gumbel, attention_gumbel_mock
        # Two encoders, one for image and vector input, one for vector input.
        self.emergency_feature_dim = self.custom_config["emergency_feature_dim"]
        if self.emergency_encoder_arch == 'mlp' or self.emergency_encoder_arch == 'attention':
            agents_state_dim = self.custom_config['status_dim']
        elif 'residual' in self.emergency_encoder_arch:
            agents_state_dim = self.custom_config[
                                   'status_dim'] + self.emergency_feature_dim * self.emergency_queue_length
        else:
            agents_state_dim = self.custom_config['status_dim'] + self.emergency_feature_dim
        status_grid_space = dict(obs=spaces.Dict({
            'agents_state': spaces.Box(low=0, high=2, shape=(agents_state_dim,), dtype=np.float32),
            'grid': spaces.Box(low=0, high=1, shape=eval(self.custom_config['grid_dim']), dtype=np.float32)
        }))
        emergency_obs_space = dict(obs=spaces.Box(low=0, high=2, shape=(self.emergency_dim,), dtype=np.float32))
        self.status_grid_encoder = BaseEncoder(self.model_arch_args['status_encoder'], status_grid_space)

        if 'mlp' in self.emergency_encoder_arch:
            actual_input_args = {'custom_model_config': {'model_arch_args': emergency_encoder_arch_args}}
            self.emergency_encoder = BaseEncoder(actual_input_args, emergency_obs_space)
        elif 'attention' in self.emergency_encoder_arch:
            emergency_encoder_arch_args['emergency_queue_length'] = model_arch_args['emergency_queue_length']
            emergency_encoder_arch_args['agents_state_dim'] = (self.custom_config['status_dim'])
            emergency_encoder_arch_args['num_heads'] = model_arch_args['num_heads']
            emergency_encoder_arch_args['attention_dim'] = model_arch_args['attention_dim'] * model_arch_args[
                'num_heads']
            emergency_encoder_arch_args['emergency_feature'] = self.emergency_feature_dim
            emergency_encoder_arch_args['with_softmax'] = 'gumbel' not in self.emergency_encoder_arch
            self.emergency_encoder = CrowdSimAttention(emergency_encoder_arch_args)
        else:
            raise ValueError(f"Emergency encoder architecture {self.emergency_encoder_arch} not supported.")
        self.emergency_selector = SlimFC(
            in_size=self.emergency_encoder.output_dim,
            out_size=model_arch_args['emergency_queue_length'],
            initializer=normc_initializer(1.0),
            activation_fn=None,
        )
        if 'gumbel' in self.emergency_encoder_arch:
            self.output_dim = self.status_grid_encoder.output_dim
        else:
            self.output_dim = self.status_grid_encoder.output_dim + self.emergency_encoder.output_dim

    def forward(self, inputs):
        status = inputs[Constants.VECTOR_STATE][..., :-self.emergency_dim]
        emergency = inputs[Constants.VECTOR_STATE][..., -self.emergency_dim:]
        batch_size = emergency.shape[0]
        if 'mlp' in self.emergency_encoder_arch:
            emergency_embedding = self.emergency_encoder(emergency)
            if 'residual' in self.emergency_encoder_arch:
                executor_obs = inputs[Constants.VECTOR_STATE]
            else:
                executor_obs = status
        else:
            emergency_embedding, weights_matrix = self.emergency_encoder(
                {Constants.VECTOR_STATE: status, 'emergency': emergency}
            )
            self.last_weight_matrix = weights_matrix
            if 'gumbel' in self.emergency_encoder_arch:
                invalid_mask = (emergency.reshape(-1, self.emergency_queue_length,
                                                  self.emergency_feature_dim) != 0).sum(
                    axis=2) == 0
                if 'mock' in self.emergency_encoder_arch:
                    # randomly generate selection result (as one hot vector)
                    selection = torch.randint(0, self.emergency_queue_length, (batch_size,))
                    # convert to one hot
                    selection = F.one_hot(selection, self.emergency_queue_length).to(torch.float32).to(emergency.device)
                else:
                    weights_matrix[invalid_mask] = -1e8
                    selection = gumbel_softmax(weights_matrix + 1e-8, temperature=0.1)
                logging.debug(f"selection: {selection[:5]}")
                selected_emergency = torch.einsum('bi,bij->bj', selection,
                                                  emergency.reshape(batch_size, -1, self.emergency_feature_dim))
                executor_obs = torch.cat((status, selected_emergency), dim=-1)
                self.last_selection = selection.argmax(axis=1)
            elif 'residual' in self.emergency_encoder_arch:
                executor_obs = inputs[Constants.VECTOR_STATE]
                self.last_selection = weights_matrix.argmax(axis=1)
            else:
                executor_obs = status
                self.last_selection = weights_matrix.argmax(axis=1)
        output = self.status_grid_encoder(
            {Constants.VECTOR_STATE: executor_obs,
             Constants.IMAGE_STATE: inputs[Constants.IMAGE_STATE]}
        )
        self.last_executor_obs = executor_obs
        if 'gumbel' not in self.emergency_encoder_arch:
            return torch.cat((emergency_embedding, output), dim=-1)
        else:
            return output
