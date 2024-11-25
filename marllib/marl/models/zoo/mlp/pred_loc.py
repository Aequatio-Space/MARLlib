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
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        BaseMLPMixin.__init__(self)
        # decide the model arch 
        self.inputs = None
        self.custom_config = model_config["custom_model_config"]
        self.model_arch_args = self.custom_config['model_arch_args']
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]
        self.local_mode = self.model_arch_args['local_mode']
        self.num_envs = self.custom_config["num_envs"] if not self.local_mode else 10
        self.activation = model_config.get("fcnet_activation")
        self.horizon = self.model_arch_args["horizon"]
        self.device = get_device()

        # encoder
        self.p_encoder = BaseEncoder(model_config, self.full_obs_space).to(self.device)
        self.vf_encoder = BaseEncoder(model_config, self.full_obs_space).to(self.device)

        self.loc_pred = LocPredEncoder(max_pos_value=600).to(self.device)
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
        # print('in foward!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print('input_dict keys:', input_dict.keys())
        # Simply convert deque content to tensor
        # if len(self.agent_x_time_list) > 0:
        #     self.agent_x_history = torch.stack(list(self.agent_x_time_list), dim=0).to(self.device)
        #     self.agent_y_history = torch.stack(list(self.agent_y_time_list), dim=0).to(self.device)
        #     print(
        #         f"self.agent_x_history shape: {self.agent_x_history.shape}, self.agent_y_history shape: {self.agent_y_history.shape}")
        #     loc_pred = self.loc_pred(self.agent_x_history, self.agent_y_history)
        #     # Shape: [num_envs, num_agents, hidden_dim]
        #     loc_pred = loc_pred.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
        #     # Shape: [num_envs, num_agents, num_agents, hidden_dim]

        #     # Directly mask out each agent's own prediction
        #     for i in range(self.n_agents):
        #         loc_pred[:, i, i, :] = 0

        #     # Reshape to concatenate all predictions
        #     final_pred = loc_pred.sum(-2)
        # else:
        #     final_pred = torch.zeros((self.num_envs, self.n_agents, self.loc_pred.hidden_dim), device=self.device)
        #     # Shape: [num_envs, num_agents, hidden_dim]

        if self._is_train:
            print("train: input_dict", input_dict.keys())
        else:
            pass
        #     final_pred = final_pred.detach()
        # input_dict['pred_other_loc'] = final_pred

        return BaseMLPMixin.forward(self, input_dict, state, seq_lens)

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return BaseMLPMixin.value_function(self)