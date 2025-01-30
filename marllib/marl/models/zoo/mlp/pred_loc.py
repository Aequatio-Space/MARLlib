import logging

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from marllib.marl.models.zoo.mlp.base_mlp import BaseMLPMixin
from marllib.marl.algos.utils.setup_utils import get_device
from marllib.marl.models.zoo.encoder import BaseEncoder, LocPredEncoder
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.typing import Dict, TensorType, List
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn
import wandb
from collections import deque
from warp_drive.utils.constants import Constants
from gym.spaces import Box
import numpy as np
from einops import rearrange

class PredLoc(TorchModelV2, nn.Module, BaseMLPMixin):
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
        state_dim = obs_space.shape[0]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        print('------------ original_space ------------')
        print(self.full_obs_space)
        original_shape = self.full_obs_space['obs']['agents_state'].shape[0]
        new_shape = original_shape + 64

        # Create new Box space
        new_box = Box(
            low=-1e+20,
            high=1e+20,
            shape=(new_shape,),
            dtype=np.float32
        )

        # Update the full_obs_space
        # self.full_obs_space['obs']['agents_state'] = new_box
        # # Handle the obs_space update
        # if hasattr(obs_space, "original_space"):
        #     obs_space.original_space['obs']['agents_state'] = new_box
        # else:
        #     obs_space['obs']['agents_state'] = new_box
        # new_obs_space = Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape= (state_dim+64,),
        #     dtype=np.float32
        # )
        # setattr(new_obs_space, "original_space", self.full_obs_space)
        # obs_space = new_obs_space

        # Initialize TorchModelV2 with updated obs_space
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        BaseMLPMixin.__init__(self)
        # decide the model arch 
        self.inputs = None
        self.custom_config = model_config["custom_model_config"]
        self.model_arch_args = self.custom_config['model_arch_args']

        self.n_agents = self.custom_config["num_agents"]
        self.local_mode = self.model_arch_args['local_mode']
        self.num_envs = self.custom_config["num_envs"] if not self.local_mode else 10
        self.activation = model_config.get("fcnet_activation")
        self.horizon = self.model_arch_args["horizon"]
        self.device = get_device()
        self.loc_pred = LocPredEncoder(max_pos_value=600).to(self.device)
        self.hidden_dim = self.loc_pred.hidden_dim
        # encoder
        logging.debug('==================================')
        logging.debug(model_config)
        logging.debug('==================================')

        self.p_encoder = BaseEncoder(model_config, self.full_obs_space).to(self.device)
        self.vf_encoder = BaseEncoder(model_config, self.full_obs_space).to(self.device)

        self.p_branch = SlimFC(
            in_size=self.p_encoder.output_dim,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None).to(self.device)

        # self.vf_encoder = nn.Sequential(*copy.deepcopy(layers))
        self.vf_branch = SlimFC(
            in_size=self.vf_encoder.output_dim,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None).to(self.device)
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
        # shape: [horizon + 1, num_envs, num_agents]
        self.agent_x_time_list = deque(maxlen=self.horizon + 1)  # for history
        self.agent_y_time_list = deque(maxlen=self.horizon + 1)  # for history
        self.current_timestep = 0
        if wandb.run is not None:
            wandb.watch(models=tuple(self.actors), log='all')

    def train(self):
        logging.debug("train is called")
        self._is_train = True

    def eval(self):
        logging.debug("eval is called")
        self._is_train = False

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        logging.debug(input_dict.keys())
        x_list = self.agent_x_time_list
        y_list = self.agent_y_time_list
        num_agents = self.n_agents
        num_envs = self.num_envs
        length_x = len(x_list)
        length_y = len(y_list)
        # new_obs = input_dict['new_obs']
        # if new_obs.shape[0] == 1:
        #     exit(0)
        #
        # if new_obs.ndim == 3:
        #     if length_x == self.horizon + 1 and length_y == self.horizon + 1:
        #         x_arr = np.array(x_list)
        #         y_arr = np.array(y_list)
        #         assert x_arr.shape == (self.horizon + 1, num_envs, num_agents)
        #         # shape: [horizon + 1, num_envs, num_agents]
        #     else:
        #         x_arr = np.zeros((self.horizon + 1, num_envs, num_agents))
        #         y_arr = np.zeros((self.horizon + 1, num_envs, num_agents))
        #
        #     x_tensor = torch.from_numpy(x_arr).to(self.device).float()
        #     y_tensor = torch.from_numpy(y_arr).to(self.device).float()
        #     agent_coords = torch.cat((x_tensor.unsqueeze(-1), y_tensor.unsqueeze(-1)), dim=-1)
        #     agent_coords = rearrange(agent_coords, 'h e a p -> (e a) h p')
        #     loc_pred = self.loc_pred(x_tensor, y_tensor)
        #     loc_pred_obs = loc_pred.unsqueeze(1).repeat(1, num_agents, 1, 1)
        #     # Shape: [num_envs, num_agents, num_agents, hidden_dim]
        #
        #     # Directly mask out each agent's own prediction
        #     for i in range(num_agents):
        #         loc_pred_obs[:, i, i, :] = 0
        #
        #     # Reshape to concatenate all predictions
        #     # Shape: [num_envs, num_agents, hidden_dim]
        #     loc_pred_obs = loc_pred_obs.sum(-2)
        # else:
        #     loc_pred_obs = torch.zeros(32, self.hidden_dim).to(self.device)
        #
        #
        # for keys in input_dict['obs']['obs'].keys():
        #     print('obs:', keys, input_dict['obs']['obs'][keys].shape)
        #
        # for keys in input_dict['obs']['state'].keys():
        #     print('state:', keys, input_dict['obs']['state'][keys].shape)
        #
        # print('obs_flat:', input_dict['obs_flat'].shape)
        #
        # if self._is_train:
        #     print("train: input_dict", input_dict.keys())
        # else:
        #     pass
        #
        # # new_obs = torch.cat([new_obs, loc_pred_obs], dim=-1)
        # input_dict['new_obs'] = new_obs
        # print('---------------------> state.shape:', state)
        # print('\n')
        # print(len(state))
        # print(input_dict['new_obs'].shape)

        return BaseMLPMixin.forward(self, input_dict, state, seq_lens)

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return BaseMLPMixin.value_function(self)