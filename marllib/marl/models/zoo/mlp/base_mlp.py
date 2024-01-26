# MIT License
import datetime
import logging
import random
from numba import njit
import numpy as np
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
from marllib.marl.models.zoo.encoder.triple_encoder import TripleHeadEncoder
from warp_drive.utils.constants import Constants

torch, nn = try_import_torch()


def argmax_to_mask(argmax_indices, num_classes=4, num_coords=2):
    # Create a mask tensor filled with zeros
    mask = torch.zeros((len(argmax_indices), num_coords * (num_classes - 1)), dtype=torch.float32)

    # Set the values in the mask based on argmax indices
    for i, index in enumerate(argmax_indices):
        if index == 0:
            # Class 0 masks all entries
            continue
        else:
            # Calculate the start index for the corresponding class in the mask
            start_index = (index - 1) * num_coords
            # Set the values in the mask for the corresponding class
            mask[i, start_index:start_index + num_coords] = 1.0

    # Reshape the mask to have dimensions (len(argmax_indices), num_coords, num_classes)
    return mask


class Predictor(nn.Module):
    def __init__(self, input_dim=22, hidden_size=64, output_dim=1):
        super(Predictor, self).__init__()
        self.activation = nn.ReLU
        self.fc = nn.Sequential(
            SlimFC(
                in_size=input_dim,
                out_size=hidden_size,
                initializer=normc_initializer(0.01),
                activation_fn=self.activation),
            SlimFC(
                in_size=hidden_size,
                out_size=hidden_size,
                initializer=normc_initializer(0.01),
                activation_fn=self.activation),
            SlimFC(
                in_size=hidden_size,
                out_size=output_dim,
                initializer=normc_initializer(0.01),
                activation_fn=None),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.fc(x)


class RandomSelector(nn.Module):
    def __init__(self, num_agents, num_actions):
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions

    def forward(self, input_obs):
        return torch.rand((input_obs.shape[0]))


class GreedySelector(nn.Module):
    def __init__(self, num_agents, num_actions):
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions

    def forward(self, input_obs):
        agent_positions = input_obs[:, self.num_agents + 2:self.num_agents + 4]
        target_positions = input_obs[:, (self.num_agents + 4) + 4 * (self.num_agents - 1):]
        distances = torch.norm(agent_positions - target_positions, dim=1)
        return distances


class BaseMLPMixin:

    def __init__(self):
        self.inputs = None
        self._features = None
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
        self.model_arch_args = self.custom_config['model_arch_args']
        if self.model_arch_args['local_mode']:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selector_type = (self.model_arch_args['selector_type'] or
                              self.custom_config["selector_type"])
        self.episode_length = 120
        self.render = self.model_arch_args['render']
        if self.render:
            self.render_file_name = self.model_arch_args['render_file_name']
        else:
            self.render_file_name = ""

        # self.count = 0
        self._is_train = False
        self.with_task_allocation = self.custom_config["with_task_allocation"]
        self.separate_encoder = self.custom_config["separate_encoder"]
        self.num_envs = self.custom_config["num_envs"]
        self.status_dim = self.custom_config["status_dim"]
        self.gen_interval = self.model_arch_args['gen_interval']
        self.emergency_feature_dim = self.custom_config["emergency_feature_dim"]
        self.emergency_count = int(((self.episode_length / self.gen_interval) - 1) * (self.n_agents - 1))
        # self.emergency_label_number = self.emergency_dim // self.emergency_feature_dim + 1
        self.emergency_mode = self.emergency_target = None
        self.emergency_indices = None

        self.last_emergency_selection = None
        self.reset_states()
        for item in ['last_emergency_mode', 'last_emergency_target']:
            setattr(self, item, None)
        # encoder
        if self.separate_encoder:
            self.vf_encoder = TripleHeadEncoder(self.custom_config, self.model_arch_args, self.full_obs_space).to(
                self.device)
            self.p_encoder = TripleHeadEncoder(self.custom_config, self.model_arch_args, self.full_obs_space).to(
                self.device)
        else:
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
        self.last_virtual_obs = None
        self.last_predicted_values = None
        self.actors = [self.p_encoder, self.p_branch]
        self.critics = [self.vf_encoder, self.vf_branch]

        self.actor_initialized_parameters = self.actor_parameters()
        if self.selector_type == "greedy":
            self.selector = GreedySelector(self.n_agents, num_outputs)
        elif self.selector_type == 'random':
            self.selector = RandomSelector(self.n_agents, num_outputs)
        elif self.selector_type == 'NN':
            self.selector = Predictor(self.full_obs_space['obs']['agents_state'].shape[0]).to(self.device)
        # Note, the final activation cannot be tanh, check.
        if self.render:
            # self.wait_time_list = torch.zeros([self.episode_length, self.n_agents], device=self.device)
            self.emergency_mode_list = np.zeros([self.episode_length, self.n_agents], dtype=np.bool_)
            self.emergency_target_list = np.zeros([self.episode_length, self.n_agents, 2], device=self.device)
            # self.wait_time_list = torch.zeros([self.episode_length, self.n_agents], device=self.device)
            self.collision_count_list = torch.zeros([self.episode_length, self.n_agents], device=self.device)
        if wandb.run is not None:
            wandb.watch(models=tuple(self.actors), log='all')


    def reset_states(self):
        self.last_emergency_selection = torch.zeros(self.n_agents, device=self.device)
        self.emergency_mode = np.zeros((self.n_agents * self.num_envs), dtype=np.bool_)
        self.emergency_indices = np.full((self.n_agents * self.num_envs), -1, dtype=np.int32)
        self.emergency_target = np.full((self.n_agents * self.num_envs, 2), -1, dtype=np.float32)
        # same mode, indices, target with "mock" for testing
        # self.mock_emergency_mode = np.zeros((self.n_agents * self.num_envs), dtype=np.bool_)
        # self.mock_emergency_indices = np.full((self.n_agents * self.num_envs), -1, dtype=np.int32)
        # self.mock_emergency_target = np.full((self.n_agents * self.num_envs, 2), -1, dtype=np.float32)
        # self.wait_time = torch.zeros((self.n_agents * self.num_envs), device=self.device, dtype=torch.int32)
        # self.collision_count = torch.zeros((self.n_agents * self.num_envs), device=self.device, dtype=torch.int32)

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
        observation = input_dict['obs']['obs']
        if isinstance(observation, dict):
            flat_inputs = {k: v.float() for k, v in observation.items()}
        else:
            flat_inputs = observation.float()
        if self.with_task_allocation:
            self.query_and_assign(flat_inputs, input_dict)
        return BaseMLPMixin.forward(self, input_dict, state, seq_lens)

    def one_time_assign(self, flat_inputs, input_dict):
        agent_observations = flat_inputs[Constants.VECTOR_STATE]
        selection_result = torch.argmax(self.selector(agent_observations), dim=1)
        obs_masks = argmax_to_mask(selection_result, num_classes=self.emergency_label_number, num_coords=2)
        agent_observations[..., self.status_dim:] *= obs_masks
        input_dict['obs']['obs']['agents_state'] = agent_observations
        self.last_emergency_selection = selection_result[:self.n_agents]

    def old_task_assign_query(self, target_coverage, emergency_xy, all_obs):
        for i, this_coverage in enumerate(target_coverage):
            # logging.debug(f"index: {i}")
            if self.mock_emergency_mode[i]:
                # check emergency coverage
                if self.mock_emergency_indices[i] != -1 and this_coverage[self.mock_emergency_indices[i]]:
                    # target is covered, reset emergency mode
                    logging.debug(f"Emergency target ({self.mock_emergency_target[i][0]},"
                                  f"{self.mock_emergency_target[i][1]}) is covered by agent")
                    self.mock_emergency_mode[i] = 0
                    self.mock_emergency_target[i] = -1
                    self.mock_emergency_indices[i] = -1
                    # self.mock_collision_count[i] = 0
                    # self.mock_wait_time[i] = 0
                else:
                    # fill in original target
                    all_obs[i][self.status_dim:self.status_dim + self.emergency_feature_dim] = torch.from_numpy(
                        self.emergency_target[i]).to(all_obs.device)
                # predicted_values[i] = -1
            else:
                valid_emergencies = this_coverage == 0
                actual_indices = np.nonzero(valid_emergencies)[0]
                valid_emergencies_xy = emergency_xy[valid_emergencies]
                num_of_emergency = len(valid_emergencies_xy)
                if num_of_emergency > 0:
                    # query predictor for new emergency target
                    queries_obs = all_obs[i].repeat(len(valid_emergencies_xy), 1).to(self.device)
                    queries_obs[...,
                    self.status_dim:self.status_dim + self.emergency_feature_dim] = torch.from_numpy(
                        valid_emergencies_xy)
                    with torch.no_grad():
                        batch_predicted_values = self.selector(queries_obs)
                    # find the min distance
                    min_index = torch.argmin(batch_predicted_values)
                    # set emergency mode
                    self.mock_emergency_mode[i] = 1
                    # set emergency target
                    self.mock_emergency_target[i] = valid_emergencies_xy[min_index]
                    self.mock_emergency_indices[i] = actual_indices[min_index]
                    # fill in emergency target
                    all_obs[i][self.status_dim:self.status_dim + self.emergency_feature_dim] = \
                        torch.from_numpy(self.mock_emergency_target[i])
                    # predicted_values[i] = batch_predicted_values[min_index]
                    # logging.debug(
                    #     f"agent selected Target:"
                    #     f"({self.mock_emergency_target[i][0].item()},{self.mock_emergency_target[i][1].item()}) "
                    #     f"with metric value {predicted_values[i]}"
                    # )
                # else:
                # predicted_values[i] = -1
        return all_obs

    def query_and_assign(self, flat_inputs, input_dict):
        if not self._is_train:
            timestep = input_dict['obs']['state'][Constants.VECTOR_STATE][..., -1][0].to(torch.int32)
            logging.debug("NN logged timestep: {}".format(timestep))
            if timestep == 0:
                for item in ['emergency_mode', 'emergency_target']:
                    setattr(self, 'last_' + item, getattr(self, item))
                # reset network mode
                self.reset_states()
            else:
                all_obs = flat_inputs[Constants.VECTOR_STATE]
                # logging.debug(f"Mode before obs: {self.emergency_mode.cpu().numpy()}")
                if torch.sum(all_obs) > 0:
                    numpy_observations = all_obs.cpu().numpy()
                    emergency_status = input_dict['obs']['state'][Constants.VECTOR_STATE][...,
                                       self.n_agents * (self.n_agents + 4):-1].reshape(-1, self.emergency_count,
                                                                                       4).cpu().numpy()
                    emergency_xy = np.stack((emergency_status[0][..., 0], emergency_status[0][..., 1]), axis=-1)
                    target_coverage = emergency_status[..., 3]
                    if self.selector_type == 'oracle':
                        # split all_obs into [num_envs] of vectors
                        all_env_obs = all_obs.reshape(-1, self.n_agents, 22)
                        for i, (env_obs, this_coverage) in enumerate(zip(all_env_obs,
                                                                         target_coverage[::self.n_agents])):
                            covered_emergency = emergency_xy[this_coverage == 1]
                            for j in range(self.n_agents):
                                actual_agent_id = i * self.n_agents + j
                                if self.emergency_target[actual_agent_id] in covered_emergency:
                                    # reset emergency mode
                                    logging.debug(
                                        f"Emergency target ({self.emergency_target[actual_agent_id][0].item()},"
                                        f"{self.emergency_target[actual_agent_id][1].item()}) in env {i} "
                                        f"is covered by agent {j}")
                                    self.emergency_mode[actual_agent_id] = 0
                                    self.emergency_target[actual_agent_id] = -1
                            # reassign the original emergency target for agents who didn't complete the task
                            for j in range(self.n_agents):
                                actual_agent_id = i * self.n_agents + j
                                if self.emergency_mode[actual_agent_id] == 1:
                                    # fill in original target
                                    all_obs[actual_agent_id][self.status_dim:
                                                             self.status_dim + self.emergency_feature_dim] = (
                                        self.emergency_target)[actual_agent_id]
                            # get environment state and calculate distance with each agent
                            valid_emergencies_xy = emergency_xy[this_coverage == 0]
                            num_of_emergency = len(valid_emergencies_xy)
                            agents_xy = env_obs[..., self.n_agents + 2: self.n_agents + 4].repeat(
                                num_of_emergency, 1).reshape(num_of_emergency, self.n_agents, 2).to(self.device)
                            matrix_emergency = valid_emergencies_xy.repeat(
                                1, self.n_agents).reshape(num_of_emergency, self.n_agents, 2)
                            distances = torch.norm(agents_xy - matrix_emergency, dim=2)
                            # for each valid emergency, find the closest agent
                            for j in range(num_of_emergency):
                                is_allocated = False
                                for agent_emergency in self.emergency_target[
                                                       i * self.n_agents: (i + 1) * self.n_agents]:
                                    if torch.all(agent_emergency == valid_emergencies_xy[j]):
                                        # this emergency is already allocated
                                        is_allocated = True
                                        break
                                if not is_allocated:
                                    query_index = torch.argsort(distances[j])
                                    for agent_id in query_index:
                                        agent_actual_index = i * self.n_agents + agent_id
                                        if self.emergency_mode[agent_actual_index] == 0:
                                            # assign emergency to this agent
                                            self.emergency_mode[agent_actual_index] = 1
                                            self.emergency_target[agent_actual_index] = valid_emergencies_xy[j]
                                            logging.debug(f"Allocated Emergency for agent {agent_id} in env {i}: "
                                                          f"{valid_emergencies_xy[j]}")
                                            # fill in emergency target
                                            all_obs[agent_actual_index,
                                            self.status_dim:self.status_dim + self.emergency_feature_dim] = (
                                                valid_emergencies_xy[j].to(all_obs.device))
                                            break
                    else:
                        # old_reference = self.old_task_assign_query(target_coverage, emergency_xy,
                        #                                            all_obs.clone().detach())
                        (query_batch, actual_emergency_indices, stop_index,
                         last_round_emergency_agents, allocation_agents) = (
                            construct_query_batch(
                                numpy_observations,
                                self.emergency_mode,
                                self.emergency_target,
                                self.emergency_indices,
                                emergency_xy,
                                target_coverage,
                                self.status_dim,
                                self.emergency_feature_dim
                            )
                        )
                        logging.debug("actual emergency indices: {}".format(actual_emergency_indices))
                        if query_batch is not None:
                            query_batch = query_batch.reshape(-1, self.status_dim + self.emergency_feature_dim)
                            inputs = torch.from_numpy(query_batch).to(self.device)
                            with torch.no_grad():
                                outputs = self.selector(inputs).cpu().numpy()
                                if len(outputs.shape) > 1:
                                    outputs = outputs.squeeze(-1)
                            all_obs = construct_final_observation(
                                numpy_observations,
                                query_batch,
                                self.emergency_target,
                                self.emergency_indices,
                                actual_emergency_indices,
                                outputs,
                                last_round_emergency_agents,
                                allocation_agents,
                                stop_index,
                                self.status_dim,
                                self.emergency_feature_dim
                            )
                            all_obs = torch.from_numpy(all_obs).to(self.device)
                            # assert torch.all(all_obs == old_reference)
                            # assert np.all(self.emergency_mode == self.mock_emergency_mode)
                            # assert np.all(self.emergency_indices == self.mock_emergency_indices)
                            # assert np.all(self.emergency_target == self.mock_emergency_target)
                input_dict['obs']['obs']['agents_state'] = all_obs
            logging.debug(f"Mode after obs: {self.emergency_mode[:self.n_agents]}")
            logging.debug(f"Actual Emergency Indices: {self.emergency_indices[:self.n_agents]}")
            if self.render:
                self.emergency_mode_list[timestep] = self.emergency_mode[:self.n_agents]
                self.emergency_target_list[timestep] = self.emergency_target[:self.n_agents]
            if timestep == self.episode_length - 1:
                if self.render:
                    # WARNING: not modified for numpy.
                    # output all rendering information as csv
                    import pandas as pd
                    # concatenate all rendering information
                    all_rendering_info = torch.cat(
                        [self.emergency_mode_list,
                         self.emergency_target_list.reshape(self.episode_length, -1),
                         ], dim=-1)
                    # convert to numpy
                    all_rendering_info = all_rendering_info.cpu().numpy()
                    # convert to pandas dataframe
                    all_rendering_info = pd.DataFrame(all_rendering_info)
                    # save to csv
                    datatime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    all_rendering_info.to_csv(f"{self.render_file_name}_{datatime_str}.csv")
            self.last_virtual_obs = input_dict['obs']['obs']['agents_state']
        else:
            try:
                input_dict['obs']['obs']['agents_state'] = input_dict['virtual_obs']
            except KeyError:
                logging.debug("No virtual obs found")

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return BaseMLPMixin.value_function(self)


@njit
def construct_query_batch(
        all_obs: np.ndarray,
        my_emergency_mode: np.ndarray,
        my_emergency: np.ndarray,
        my_emergency_index: np.ndarray,
        emergency_xy: np.ndarray,
        target_coverage: np.ndarray,
        status_dim: int,
        emergency_feature_dim: int,
):
    """
    set emergency_mode according to coverage status and return a new batch of emergency mode status
    find valid emergency and put them into the query list (for later inference of predictor)
    return:
    new_emergency_mode
    query_batch
    agent_bounds_for_emergency
    """
    query_obs_list = []
    query_index_list = []
    actual_emergency_index_list = []
    allocation_agent_list = []
    last_round_emergency = my_emergency_mode.copy()

    start_index = 0
    for i, this_coverage in enumerate(target_coverage):
        #     # logging.debug(f"index: {i}")
        if my_emergency_mode[i]:
            # check emergency coverage
            if this_coverage[my_emergency_index[i]]:
                # target is covered, reset emergency mode
                # logging.debug("my_emergency_index: {}".format(my_emergency_index[i]))
                my_emergency_mode[i] = 0
                my_emergency[i] = -1
                my_emergency_index[i] = -1
        else:
            valid_emergencies = this_coverage == 0
            actual_emergency_indices = np.nonzero(valid_emergencies)[0]
            valid_emergencies_xy = emergency_xy[valid_emergencies]
            num_of_emergency = len(valid_emergencies_xy)
            if num_of_emergency > 0:
                # query predictor for new emergency target
                for emergency in valid_emergencies_xy:
                    query_obs = all_obs[i].copy()
                    query_obs[status_dim:status_dim + emergency_feature_dim] = emergency
                    query_obs_list.extend(query_obs)
                start_index += num_of_emergency
                query_index_list.append(start_index)
                actual_emergency_index_list.extend(actual_emergency_indices)
                allocation_agent_list.append(i)
                my_emergency_mode[i] = 1
    last_round_emergency = np.nonzero(my_emergency_mode & last_round_emergency)[0]
    # concatenate all queries
    if len(query_obs_list) > 0:
        return (np.array(query_obs_list), np.array(actual_emergency_index_list),
                np.array(query_index_list), last_round_emergency, np.array(allocation_agent_list))
    else:
        return (None,) * 5


@njit
def construct_final_observation(
        all_obs: np.ndarray,
        query_batch: np.ndarray,
        my_emergency_target: np.ndarray,
        my_emergency_indices: np.ndarray,
        actual_emergency_indices: np.ndarray,
        predicted_values: np.ndarray,
        last_round_emergency_mode: np.ndarray,
        allocation_agent_list: np.ndarray,
        stop_index: np.ndarray,
        status_dim: int,
        emergency_feature_dim: int,
):
    start_index = 0
    for stop, emergency_agent_index in zip(stop_index, allocation_agent_list):
        argmin_index = np.argmin(predicted_values[start_index:stop])
        offset = start_index + argmin_index
        all_obs[emergency_agent_index] = query_batch[offset]
        my_emergency_indices[emergency_agent_index] = actual_emergency_indices[offset]
        my_emergency_target[emergency_agent_index] = (query_batch[offset]
        [status_dim:status_dim + emergency_feature_dim])
        start_index = stop
    for index in last_round_emergency_mode:
        all_obs[index][status_dim:status_dim + emergency_feature_dim] = my_emergency_target[index]
    return all_obs


class CrowdSimAttention(TorchModelV2, nn.Module, BaseMLPMixin):
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


def log_emergency_target_draft():
    pass
    # if wandb.run is not None and len(predicted_values) >= self.n_agents:
    #     # log agent allocation status
    #     for i in range(self.n_agents):
    #         wandb.log({"agent_{}_emergency_mode".format(i): emergency_mode[i].item()})
    #         wandb.log({"agent_{}_emergency_target".format(i): predicted_values[i].item()})
    # self.last_predicted_values = predicted_values