import numpy as np
from gym import spaces
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils import try_import_torch

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
        self.activation = self.custom_config.get("fcnet_activation")
        self.emergency_dim = self.custom_config['emergency_dim']
        # Two encoders, one for image and vector input, one for vector input.
        status_grid_space = dict(obs=spaces.Dict({
            'agents_state': spaces.Box(low=0, high=2, shape=(self.custom_config['status_dim'],), dtype=np.float32),
            'grid': spaces.Box(low=0, high=1, shape=eval(self.custom_config['grid_dim']), dtype=np.float32)
        }))
        emergency_obs_space = dict(obs=spaces.Box(low=0, high=2, shape=(self.emergency_dim,), dtype=np.float32))
        self.status_grid_encoder = BaseEncoder(self.model_arch_args['status_encoder'], status_grid_space)
        self.emergency_encoder = BaseEncoder(self.model_arch_args['emergency_encoder'], emergency_obs_space)
        self.output_dim = self.emergency_encoder.output_dim + self.status_grid_encoder.output_dim
        # Merge the three branches
        # self.merge_branch = SlimFC(
        #     in_size=full_feature_output,
        #     out_size=self.custom_config["hidden_state_size"],
        #     initializer=normc_initializer(1.0),
        #     activation_fn=self.activation,
        # )

    def forward(self, inputs):
        status = inputs[Constants.VECTOR_STATE][..., :-self.emergency_dim]
        emergency = inputs[Constants.VECTOR_STATE][..., -self.emergency_dim:]
        output = self.status_grid_encoder({Constants.VECTOR_STATE: status,
                                           Constants.IMAGE_STATE: inputs[Constants.IMAGE_STATE]})
        emergency = self.emergency_encoder(emergency)
        status_embedding, grid_embedding = output[..., :self.status_grid_encoder.dims[0]], \
            output[..., self.status_grid_encoder.dims[0]:]
        # x = self.merge_branch(torch.cat((status_embedding, emergency, grid_embedding), dim=-1))
        return torch.cat((status_embedding, emergency, grid_embedding), dim=-1)
