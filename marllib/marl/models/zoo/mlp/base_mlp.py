# MIT License
import logging
import os
import pprint
from collections import deque
from functools import reduce
from typing import Tuple
import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd
import wandb
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder
from marllib.marl.models.zoo.encoder.triple_encoder import TripleHeadEncoder
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.typing import Dict, TensorType, List
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds
from torch.distributions import Categorical

from warp_drive.utils.common import get_project_root
from warp_drive.utils.constants import Constants

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

torch, nn = try_import_torch()


def generate_identity_matrices(n, m):
    # Generate n identity matrices of size m
    identity_matrices = [np.eye(m) for _ in range(n)]
    # Stack the identity matrices horizontally
    result = np.hstack(identity_matrices)
    return result


def find_indices_for_cost(cost_array, matrix, atol):
    indices_list = []
    for row, cost in zip(matrix, cost_array):
        # Find indices where the cost is equal or slightly larger than the corresponding row of the matrix
        logging.debug(f"reference cost: {cost}, row: {row}")
        indices = np.nonzero(np.isclose(row, cost, atol=atol))[0]
        logging.debug("indices: {}".format(indices))
        indices_list.append(indices)
    return indices_list


def generate_agent_matrix(num_of_agents, num_of_tasks):
    matrix = np.zeros((num_of_agents, num_of_agents * num_of_tasks), dtype=int)
    col_indices = np.arange(0, num_of_agents * num_of_tasks)
    row_indices = np.repeat(np.arange(0, num_of_agents), repeats=num_of_tasks)
    # Use NumPy broadcasting to set the values efficiently
    matrix[row_indices, col_indices] = 1
    return matrix


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
                activation_fn=nn.Sigmoid),
            # activation_fn=None),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.fc(x)


class AgentSelector(nn.Module):
    def __init__(self, input_dim, hidden_size, num_agents):
        super(AgentSelector, self).__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim
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
                out_size=num_agents,
                initializer=normc_initializer(0.01),
                activation_fn=None),
        )
        # softmax layer

    def forward(self, input_obs, invalid_mask=None):
        input_obs = self.fc(input_obs)
        if invalid_mask is not None:
            input_obs -= invalid_mask * (FLOAT_MAX / 2)
        action_scores = torch.nn.functional.softmax(input_obs, dim=1)
        return action_scores


class RandomSelector(nn.Module):
    def __init__(self, num_agents, num_actions):
        super(RandomSelector, self).__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions

    def forward(self, input_obs):
        return torch.rand((input_obs.shape[0]))


class GreedySelector(nn.Module):
    def __init__(self, num_agents, num_actions):
        super(GreedySelector, self).__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions

    def forward(self, input_obs):
        agent_positions = input_obs[:, self.num_agents + 2:self.num_agents + 4]
        target_positions = input_obs[:, (self.num_agents + 4) + 4 * (self.num_agents - 1):]
        distances = torch.norm(agent_positions - target_positions, dim=1)
        return distances


class GreedyAgentSelector(nn.Module):
    def __init__(self, input_dim, num_agents):
        super(GreedyAgentSelector, self).__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim

    def forward(self, input_obs, invalid_mask):
        agent_positions = input_obs[:, :-2].reshape(-1, self.num_agents, 3)[..., :2]
        target_positions = input_obs[:, -2:].repeat(self.num_agents, 1).reshape(-1, self.num_agents, 2)
        distances = torch.norm(agent_positions - target_positions, dim=-1)
        distances += (FLOAT_MAX - 5) * invalid_mask
        return torch.nn.functional.one_hot(torch.argmin(distances, dim=-1), self.num_agents)


class RandomAgentSelector(nn.Module):
    def __init__(self, input_dim, num_agents):
        super(RandomAgentSelector, self).__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim

    def forward(self, input_obs):
        return torch.rand((input_obs.shape[0], self.num_agents))


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
        self.assignment_sample_batches = []
        self.inputs = None
        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]
        self.activation = model_config.get("fcnet_activation")
        self.model_arch_args = self.custom_config['model_arch_args']
        logging.debug(f"CrowdSimNet model_arch_args: {pprint.pformat(self.model_arch_args)}")
        if self.model_arch_args['local_mode']:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selector_type = (self.model_arch_args['selector_type'] or
                              self.custom_config["selector_type"])
        self.episode_length = 120
        self.switch_step = self.model_arch_args['switch_step']
        self.step_count = 0
        self.local_mode = self.model_arch_args['local_mode']
        self.render = self.model_arch_args['render']
        if self.render:
            self.render_file_name = self.model_arch_args['render_file_name']
        else:
            self.render_file_name = ""

        # self.count = 0
        self._is_train = False
        self.with_task_allocation = self.custom_config["with_task_allocation"]
        self.separate_encoder = self.custom_config["separate_encoder"]
        self.num_envs = self.custom_config["num_envs"] if not self.local_mode else 10
        self.status_dim = self.custom_config["status_dim"]
        self.gen_interval = self.model_arch_args['gen_interval']
        self.emergency_threshold = self.model_arch_args['emergency_threshold']
        self.tolerance = self.model_arch_args['tolerance']
        self.dataset_name = self.model_arch_args['dataset']
        self.emergency_feature_dim = self.custom_config["emergency_feature_dim"]
        self.rl_update_interval = max(1, self.num_envs // 10)
        self.train_count = 0
        self.look_ahead = True
        # self.emergency_queue_length = 5
        self.emergency_queue_length = self.model_arch_args['emergency_queue_length']
        emergency_path_name = os.path.join(get_project_root(), 'datasets',
                                           self.dataset_name, 'emergency_time_loc_0900_0930.csv')
        if os.path.exists(emergency_path_name):
            self.unique_emergencies = pd.read_csv(emergency_path_name)
            self.emergency_count = len(self.unique_emergencies)
        else:
            self.unique_emergencies = None
            self.emergency_count = int(((self.episode_length / self.gen_interval) - 1) * (self.n_agents - 1))
        # self.emergency_label_number = self.emergency_dim // self.emergency_feature_dim + 1
        self.emergency_mode = self.emergency_target = self.emergency_queue = self.emergency_indices = None
        self.with_programming_optimization = self.model_arch_args['with_programming_optimization']
        self.one_agent_multi_task = self.model_arch_args['one_agent_multi_task']
        self.last_emergency_selection = self.last_emergency_indices = None
        self.last_rl_transitions = [[] for _ in range(self.num_envs)]
        self.reset_states()
        self.last_emergency_queue_length = self.last_emergency_mode = self.last_emergency_targets = None
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
            input_dim = self.full_obs_space['obs']['agents_state'].shape[0]
            self.selector = Predictor(input_dim).to(self.device)
        elif self.selector_type == 'RL':
            self.selector = AgentSelector(self.n_agents * 3 + 2, 64, self.n_agents).to(
                self.device)
            # self.selector = GreedyAgentSelector(self.n_agents * 3 + 2, self.n_agents).to(
            #     self.device)
            # self.selector = RandomAgentSelector(self.n_agents * 3 + 2, self.n_agents).to(
            #     self.device)
        # Note, the final activation cannot be tanh, check.
        if self.render:
            # self.wait_time_list = torch.zeros([self.episode_length, self.n_agents], device=self.device)
            self.emergency_mode_list = np.zeros([self.episode_length, self.n_agents], dtype=np.bool_)
            self.emergency_target_list = np.zeros([self.episode_length, self.n_agents, 2], dtype=np.float32)
            self.emergency_queue_list = np.zeros([self.episode_length, self.n_agents], dtype=np.int32)
            # self.wait_time_list = torch.zeros([self.episode_length, self.n_agents], device=self.device)
            # self.collision_count_list = torch.zeros([self.episode_length, self.n_agents])
        if wandb.run is not None:
            wandb.watch(models=tuple(self.actors), log='all')

    def reset_states(self):
        self.last_emergency_selection = torch.zeros(self.n_agents, device=self.device)
        self.emergency_mode = np.zeros((self.n_agents * self.num_envs), dtype=np.bool_)
        self.emergency_indices = np.full((self.n_agents * self.num_envs), -1, dtype=np.int32)
        self.emergency_target = np.full((self.n_agents * self.num_envs, 2), -1, dtype=np.float32)
        self.emergency_queue = [deque() for _ in range(self.n_agents * self.num_envs)]
        # same mode, indices, target with "mock" for testing
        self.mock_emergency_mode = np.zeros((self.n_agents * self.num_envs), dtype=np.bool_)
        self.mock_emergency_indices = np.full((self.n_agents * self.num_envs), -1, dtype=np.int32)
        self.mock_emergency_target = np.full((self.n_agents * self.num_envs, 2), -1, dtype=np.float32)
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
                self.last_emergency_queue_length = [len(q) for q in self.emergency_queue[:self.n_agents]]
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
                        self.oracle_assignment(all_obs, emergency_xy, target_coverage,
                                               self.emergency_mode, self.emergency_target,
                                               self.emergency_indices)
                    elif self.selector_type == 'RL':
                        self.rl_assignment(all_obs, emergency_xy, target_coverage, self.emergency_queue, self.n_agents)
                        # old_reference = self.oracle_assignment(all_obs.clone().detach(), emergency_xy, target_coverage,
                        #                        self.mock_emergency_mode, self.mock_emergency_target,
                        #                        self.mock_emergency_indices)
                        # assert torch.all(all_obs == old_reference)
                        # assert np.all(self.emergency_mode == self.mock_emergency_mode)
                        # assert np.all(self.emergency_indices == self.mock_emergency_indices)
                        # assert np.all(self.emergency_target == self.mock_emergency_target)
                    else:
                        (query_batch, actual_emergency_indices, stop_index,
                         last_round_emergency_agents, allocation_agents) = (
                            construct_query_batch(
                                numpy_observations,
                                self.emergency_mode,
                                self.emergency_target,
                                self.emergency_queue,
                                self.emergency_indices,
                                emergency_xy,
                                target_coverage,
                                self.status_dim,
                                self.emergency_feature_dim,
                                self.n_agents,
                                self.emergency_queue_length
                            )
                        )
                        logging.debug("actual emergency indices: {}".format(actual_emergency_indices))
                        logging.debug("last round emergency agents: {}".format(last_round_emergency_agents))
                        logging.debug("agents about to allocate: {}".format(allocation_agents))
                        if len(last_round_emergency_agents) == 0:
                            last_round_emergency_agents = np.array([-1], dtype=np.int32)
                        if len(query_batch) > 0:
                            query_batch = query_batch.reshape(-1, self.status_dim + self.emergency_feature_dim)
                            assert len(actual_emergency_indices) == len(query_batch)
                            assert len(stop_index) == len(allocation_agents)
                            inputs = torch.from_numpy(query_batch).to(self.device)
                            with torch.no_grad():
                                outputs = self.selector(inputs).cpu().numpy()
                                if len(outputs.shape) > 1:
                                    outputs = outputs.squeeze(-1)
                            if self.with_programming_optimization:
                                agent_envs = allocation_agents // self.n_agents
                                agent_stop_index = np.where(np.diff(agent_envs) != 0)[0] + 1
                                env_stop_index = stop_index[agent_stop_index - 1]
                                agent_nums = np.unique(agent_envs, return_counts=True)[1]
                                selected_emergencies = np.full(len(allocation_agents), -1, dtype=np.int32)
                                if len(selected_emergencies) > 0:
                                    all_cost_matrix = [matrix.reshape(my_agent_num, -1) for
                                                       my_agent_num, matrix in
                                                       zip(agent_nums, np.split(outputs, env_stop_index))]
                                    for matrix, cur_index in zip(all_cost_matrix, [0] + agent_stop_index.tolist()):
                                        row_ind, col_ind = linear_sum_assignment(matrix)
                                        logging.debug("Cost Function")
                                        logging.debug(matrix)
                                        mask = self.emergency_mode[
                                            allocation_agents[cur_index:cur_index + len(row_ind)]]
                                        selected_emergencies[cur_index + row_ind] = np.where(mask, -1, col_ind)
                                        valid_switch = self.selector_type != 'NN' or self.step_count > self.switch_step
                                        if self.one_agent_multi_task and valid_switch:
                                            target_costs = matrix[row_ind, col_ind]
                                            target_indices = find_indices_for_cost(target_costs, matrix,
                                                                                   atol=self.tolerance)
                                            for first_id, target_index, agent_index, agent_emergency in (
                                                    zip(col_ind, target_indices, row_ind, mask)):
                                                my_queue = self.emergency_queue[
                                                    allocation_agents[cur_index + agent_index]]
                                                if agent_emergency:
                                                    target_to_queue = target_index
                                                else:
                                                    target_to_queue = np.delete(target_index,
                                                                                np.where(target_index == first_id))
                                                for emergency_index in target_to_queue:
                                                    if len(my_queue) < self.emergency_queue_length:
                                                        my_queue.append(
                                                            (actual_emergency_indices[emergency_index]))
                                                    else:
                                                        break
                                else:
                                    allocation_agents = actual_emergency_indices = \
                                        selected_emergencies = np.array([-1], dtype=np.int32)
                            else:
                                selected_emergencies = np.array([], dtype=np.int32)
                        else:
                            outputs = np.array([-1], dtype=np.float32)
                            query_batch = np.full_like(all_obs[0].cpu().numpy(), -1)
                            allocation_agents = actual_emergency_indices = selected_emergencies = np.array([-1],
                                                                                                           dtype=np.int32)
                        all_obs = construct_final_observation(
                            numpy_observations,
                            query_batch,
                            self.emergency_mode,
                            self.emergency_target,
                            self.emergency_indices,
                            actual_emergency_indices,
                            selected_emergencies,
                            outputs,
                            last_round_emergency_agents,
                            allocation_agents,
                            stop_index,
                            self.status_dim,
                            self.emergency_feature_dim
                        )
                        all_obs = torch.from_numpy(all_obs).to(self.device)
                input_dict['obs']['obs']['agents_state'] = all_obs
            logging.debug(f"Mode after obs: {self.emergency_mode[:self.n_agents]}")
            logging.debug(f"Agent Emergency Indices: {self.emergency_indices[:16]}")
            logging.debug(f"Emergency Queue: {self.emergency_queue[:16]}")
            logging.debug(f"Emergency Queue Length: {[len(q) for q in self.emergency_queue[:16]]}")
            if self.render:
                if self.selector_type == 'RL':
                    for i, queue in enumerate(self.emergency_queue[:self.n_agents]):
                        if len(queue) > 0:
                            self.emergency_target_list[timestep][i] = queue[0]
                        else:
                            self.emergency_target_list[timestep][i] = -1
                else:
                    self.emergency_target_list[timestep] = self.emergency_target[:self.n_agents]
                self.emergency_mode_list[timestep] = self.emergency_mode[:self.n_agents]
                self.emergency_queue_list[timestep] = [len(q) for q in self.emergency_queue[:self.n_agents]]
            if timestep == self.episode_length - 1:
                if self.render:
                    # WARNING: not modified for numpy.
                    # output all rendering information as csv
                    import pandas as pd
                    from datetime import datetime
                    # concatenate all rendering information
                    all_rendering_info = np.concatenate(
                        [self.emergency_mode_list,
                         self.emergency_queue_list,
                         self.emergency_target_list.reshape(self.episode_length, -1),
                         ], axis=-1)
                    # convert to pandas dataframe
                    all_rendering_info = pd.DataFrame(all_rendering_info)
                    # save to csv
                    datatime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    all_rendering_info.to_csv(f"{self.render_file_name}_{datatime_str}.csv")
                self.step_count += self.episode_length * self.num_envs
            self.last_virtual_obs = input_dict['obs']['obs']['agents_state']
        else:
            try:
                input_dict['obs']['obs']['agents_state'] = input_dict['virtual_obs']
            except KeyError:
                logging.debug("No virtual obs found")

    def milp_allocate(self, actual_emergency_indices, agent_nums, agent_stop_index, allocation_agents, env_stop_index,
                      outputs, selected_emergencies):
        flatten_costs = np.split(outputs, env_stop_index)
        for flatten_cost, num_of_agents, cur_index, actual_agents in (
                zip(flatten_costs, agent_nums, [0] + agent_stop_index.tolist(),
                    np.split(allocation_agents, agent_stop_index))):
            num_of_tasks = len(flatten_cost) // num_of_agents
            logging.debug(f"num_of_agents: {num_of_agents}, num_of_tasks: {num_of_tasks}")
            # Constraint 1: each task should be assigned to 1 agent.
            ub = np.ones(num_of_tasks)
            lb = np.ones(num_of_tasks)
            constraint_mat = generate_identity_matrices(num_of_agents, num_of_tasks)
            task_constraint = LinearConstraint(constraint_mat, lb, ub)
            # Constraint 2: each agent can have 1~5 tasks
            # ub = np.zeros(num_of_agents)
            # for j in range(num_of_agents):
            #     ub[j] = (self.emergency_queue_length -
            #              len(self.emergency_queue[j + cur_index]) +
            #              ~self.emergency_mode[j + cur_index])
            # logging.debug("Agent Capacity: {}".format(ub))
            # lb = np.ones(num_of_agents)
            # constraint_mat = generate_agent_matrix(num_of_agents, num_of_tasks)
            # agent_constraint = LinearConstraint(constraint_mat, lb, ub)
            integrality = np.ones(num_of_agents * num_of_tasks)
            bounds = Bounds(0, 1)
            result = milp(flatten_cost, integrality=integrality, bounds=bounds,
                          constraints=[task_constraint])
            if result['success']:
                logging.debug("Success to solve the optimization problem")
                allocate_result = result['x'].reshape(num_of_agents, num_of_tasks).astype(
                    int)
                first_to_assign, remain_to_assign = (
                    separate_first_task(np.nonzero(allocate_result)))
                row_ind, col_ind = first_to_assign
                selected_emergencies[cur_index + row_ind] = col_ind
                for agent_id, emergency_id in zip(*remain_to_assign):
                    self.emergency_queue[actual_agents[agent_id]].append(
                        actual_emergency_indices[emergency_id])
            else:
                logging.debug("Failed to solve the optimization problem, "
                              "No allocation is made")

    def rl_assignment(self, all_obs, emergency_xy, target_coverage, my_emergency_queue, n_agents):
        all_env_obs = all_obs.clone().detach().reshape(-1, n_agents, self.status_dim + self.emergency_feature_dim)
        env_target_coverage = target_coverage[::n_agents]
        for i, this_coverage in enumerate(env_target_coverage):
            covered_emergency = this_coverage == 1
            offset = i * n_agents
            for j in range(n_agents):
                actual_agent_id = offset + j
                my_queue = my_emergency_queue[actual_agent_id]
                while len(my_queue) > 0 and covered_emergency[my_queue[0]]:
                    logging.debug(f"Emergency {my_queue[0]} is covered by agent {actual_agent_id}")
                    my_queue.popleft()

        self.do_assignment(all_env_obs, emergency_xy, env_target_coverage, my_emergency_queue, n_agents)

        self.output_assignment_result(all_obs, emergency_xy, my_emergency_queue)

    def do_assignment(self, all_env_obs, emergency_xy, env_target_coverage, my_emergency_queue, n_agents):
        for i, (env_obs, this_coverage) in enumerate(zip(all_env_obs, env_target_coverage)):
            offset = i * n_agents
            valid_emergencies = this_coverage == 0
            # convert all queue entries in an env into a list
            env_queues = [list(q) for q in my_emergency_queue[offset:offset + n_agents]]
            agents_pos = env_obs[:, n_agents + 2: n_agents + 4]
            if self.look_ahead:
                for j, my_queue in enumerate(env_queues):
                    if len(my_queue) > 0:
                        agents_pos[j] = torch.from_numpy(emergency_xy[my_queue[-1]])
                        logging.debug("Replace agent position with emergency position")
            agents_queue_len = torch.tensor([len(q) for q in env_queues]).unsqueeze(-1).to(self.device)
            single_invalid_mask = agents_queue_len.squeeze(-1) >= self.emergency_queue_length
            if torch.all(single_invalid_mask):
                logging.debug("All agents are full, no assignment is made")
                continue
            # unwrap list of list, remove emergencies in agents queue.
            additional_emergencies = [item for sublist in env_queues for item in sublist]
            logging.debug(f"Additional Emergencies to remove: {additional_emergencies}")
            valid_emergencies[additional_emergencies] = False
            emergencies = emergency_xy[valid_emergencies]
            logging.debug(f"Valid Emergencies: {emergencies}")
            received_tasks = len(emergencies)
            if received_tasks > 0:
                logging.debug(f"Valid Emergencies: {emergencies}")
                actual_emergency_indices = np.nonzero(valid_emergencies)[0]
                obs_list, action_list, reward_list = [], [], []
                for k, emergency in enumerate(emergencies):
                    if torch.all(single_invalid_mask):
                        logging.debug("All agents are full, no further assignment will be made")
                        break
                    selector_obs = torch.cat([agents_pos, agents_queue_len], dim=1).flatten()
                    final_obs = torch.cat([selector_obs, torch.from_numpy(emergency).to(self.device)])
                    final_obs = final_obs.unsqueeze(0)
                    obs_list.append(final_obs.cpu().numpy())
                    probs = self.selector(final_obs, single_invalid_mask)
                    logging.debug("Probs: {}".format(probs))
                    action_dists: Categorical = Categorical(probs=probs)
                    action = int(action_dists.sample().cpu().numpy())
                    action_list.append(action)
                    # reward = -np.linalg.norm(agents_pos[actions].cpu().numpy() - emergencies, axis=1)
                    agent_id = offset + action
                    current_queue = my_emergency_queue[agent_id]
                    if len(current_queue) < self.emergency_queue_length:
                        current_queue.append(actual_emergency_indices[k])
                        agents_queue_len[action] += 1
                        single_invalid_mask[action] = agents_queue_len[action] >= self.emergency_queue_length
                        reward_list.append(-1.0)
                    else:
                        logging.debug(f"Agent {agent_id} is full, no assignment is made")
                        reward_list.append(-2.0)
                if len(obs_list) > 0:
                    self.last_rl_transitions[i].append(
                        SampleBatch(
                            {
                                SampleBatch.REWARDS: np.array(reward_list, dtype=np.float32),
                                SampleBatch.OBS: np.vstack(obs_list),
                                SampleBatch.ACTIONS: np.array(action_list, dtype=np.int32),
                            }
                        )
                    )

    def output_assignment_result(self, all_obs, emergency_xy, my_emergency_queue):
        for i, emergency_queue in enumerate(my_emergency_queue):
            if len(emergency_queue) > 0:
                all_obs[i][self.status_dim:self.status_dim + self.emergency_feature_dim] = \
                    torch.from_numpy(emergency_xy[emergency_queue[0]])
                self.emergency_indices[i] = emergency_queue[0]
                self.emergency_target[i] = emergency_xy[emergency_queue[0]]
                self.emergency_mode[i] = 1
            else:
                self.emergency_indices[i] = -1
                self.emergency_target[i] = -1
                self.emergency_mode[i] = 0

    def oracle_assignment(self, all_obs, emergency_xy, target_coverage,
                          emergency_mode, emergency_target, emergency_indices):
        all_env_obs = all_obs.reshape(-1, self.n_agents, 22)
        for i, (env_obs, this_coverage) in enumerate(zip(all_env_obs,
                                                         target_coverage[::self.n_agents])):
            covered_emergency = emergency_xy[this_coverage == 1]
            offset = i * self.n_agents
            last_round_emergency = emergency_mode[offset:offset + self.n_agents]
            for j in range(self.n_agents):
                actual_agent_id = offset + j
                if emergency_target[actual_agent_id] in covered_emergency:
                    # reset emergency mode
                    logging.debug(
                        f"Emergency target ({emergency_target[actual_agent_id][0]},"
                        f"{emergency_target[actual_agent_id][1]}) in env {i} "
                        f"is covered by agent {j}")
                    emergency_mode[actual_agent_id] = 0
                    emergency_target[actual_agent_id] = -1
                    emergency_indices[actual_agent_id] = -1
            # reassign the original emergency target for agents who didn't complete the task
            for j in range(self.n_agents):
                actual_agent_id = offset + j
                if emergency_mode[actual_agent_id] == 1:
                    # fill in original target
                    all_obs[actual_agent_id][self.status_dim:
                                             self.status_dim + self.emergency_feature_dim] = \
                        torch.from_numpy(emergency_target[actual_agent_id])
            # get environment state and calculate distance with each agent
            valid_emergencies = this_coverage == 0
            current_emergency_agents = np.nonzero(last_round_emergency)[0]
            # temporary measure, leave the valid_emergencies untouched and let past logic handle
            # duplicate assignment.
            if self.with_programming_optimization and len(current_emergency_agents) > 0:
                this_emergency_indices = emergency_indices[current_emergency_agents + offset]
                valid_emergencies[np.delete(this_emergency_indices, np.where(this_emergency_indices == -1))] = False
            valid_emergencies_xy = emergency_xy[valid_emergencies]
            actual_emergency_indices = np.nonzero(valid_emergencies)[0]
            num_of_emergency = len(valid_emergencies_xy)
            if num_of_emergency > 0:
                available_agents = np.nonzero(~last_round_emergency)[0]
                available_agents_number = len(available_agents)
                agents_xy = env_obs[available_agents, self.n_agents + 2: self.n_agents + 4].repeat(
                    num_of_emergency, 1).reshape(num_of_emergency, available_agents_number, 2).cpu()
                matrix_emergency = np.tile(valid_emergencies_xy, reps=available_agents_number).reshape(
                    num_of_emergency, available_agents_number, 2)
                distances = np.linalg.norm(agents_xy - matrix_emergency, axis=2)
                # for each valid emergency, find the closest agent
                if self.with_programming_optimization:
                    row_ind, col_ind = linear_sum_assignment(distances.T)
                    logging.debug("Oracle Cost Function")
                    logging.debug(distances)
                    actual_agent_indices = offset + available_agents[row_ind]
                    emergency_mode[actual_agent_indices] = 1
                    selected_emergencies = valid_emergencies_xy[col_ind]
                    emergency_target[actual_agent_indices] = selected_emergencies
                    emergency_indices[actual_agent_indices] = actual_emergency_indices[col_ind]
                    all_obs[actual_agent_indices,
                    self.status_dim:self.status_dim + self.emergency_feature_dim] = \
                        torch.from_numpy(selected_emergencies).to(self.device)
                else:
                    for j in range(num_of_emergency):
                        is_allocated = False
                        for agent_emergency in emergency_target[
                                               i * self.n_agents: (i + 1) * self.n_agents]:
                            if np.all(agent_emergency == valid_emergencies_xy[j]):
                                # this emergency is already allocated
                                is_allocated = True
                                break
                        if not is_allocated:
                            query_index = np.argsort(distances[j])
                            for agent_id in query_index:
                                agent_actual_index = offset + agent_id
                                if emergency_mode[agent_actual_index] == 0:
                                    # assign emergency to this agent
                                    emergency_mode[agent_actual_index] = 1
                                    emergency_target[agent_actual_index] = valid_emergencies_xy[j]
                                    emergency_indices[agent_actual_index] = actual_emergency_indices[j]
                                    logging.debug(f"Allocated Emergency for agent {agent_id} in env {i}: "
                                                  f"{valid_emergencies_xy[j]}")
                                    # fill in emergency target
                                    all_obs[agent_actual_index,
                                    self.status_dim:self.status_dim + self.emergency_feature_dim] = \
                                        torch.from_numpy(valid_emergencies_xy[j])
                                    break
        return all_obs

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return BaseMLPMixin.value_function(self)


# @njit
def construct_query_batch(
        all_obs: np.ndarray,
        my_emergency_mode: np.ndarray,
        my_emergency_target: np.ndarray,
        my_emergency_queue: list[deque],
        my_emergency_index: np.ndarray,
        emergency_xy: np.ndarray,
        target_coverage: np.ndarray,
        status_dim: int,
        emergency_feature_dim: int,
        n_agents: int,
        max_queue_length: int,
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
        my_queue: deque = my_emergency_queue[i]
        #     # logging.debug(f"index: {i}")
        if my_emergency_mode[i] and this_coverage[my_emergency_index[i]]:
            # target is covered, reset emergency mode
            # logging.debug("my_emergency_index: {}".format(my_emergency_index[i]))
            if len(my_queue) == 0:
                my_emergency_mode[i] = 0
                my_emergency_target[i] = -1
                my_emergency_index[i] = -1
            else:
                new_emergency = my_queue.popleft()
                while this_coverage[new_emergency] and len(my_queue) > 0:
                    new_emergency = my_queue.popleft()
                my_emergency_index[i] = new_emergency
                logging.debug(f"Selecting from queue: {my_emergency_index[i]}")
                my_emergency_target[i] = emergency_xy[my_emergency_index[i]]
                logging.debug(f"Emergency target ({my_emergency_target[i][0]},{my_emergency_target[i][1]}) "
                              f"is allocated to agent {i}")
    last_round_emergency = my_emergency_mode & last_round_emergency

    for i, this_coverage in enumerate(target_coverage):
        my_queue: deque = my_emergency_queue[i]
        if not my_emergency_mode[i]:
            env_num = i // n_agents
            offset = env_num * n_agents
            valid_emergencies = this_coverage == 0
            # convert all queue entries in an env into a list
            env_queues = [list(q) for q in my_emergency_queue[offset:offset + n_agents]]
            # unwrap list of list, remove emergencies in agents queue.
            additional_emergencies = [item for sublist in env_queues for item in sublist]
            valid_emergencies[additional_emergencies] = False
            # set emergencies handled by other agents as invalid too
            current_emergency_agents = np.nonzero(last_round_emergency[offset:offset + n_agents])[0]
            if len(current_emergency_agents) > 0:
                emergency_indices = my_emergency_index[offset + current_emergency_agents]
                valid_emergencies[np.delete(emergency_indices, np.where(emergency_indices == -1)[0])] = False
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
    # concatenate all queries
    return (np.array(query_obs_list), np.array(actual_emergency_index_list),
            np.array(query_index_list), np.nonzero(last_round_emergency)[0],
            np.array(allocation_agent_list))


# @njit
def construct_final_observation(
        all_obs: np.ndarray,
        query_batch: np.ndarray,
        my_emergency_mode: np.ndarray,
        my_emergency_target: np.ndarray,
        my_emergency_indices: np.ndarray,
        actual_emergency_indices: np.ndarray,
        selected_emergencies: np.ndarray,
        predicted_values: np.ndarray,
        last_round_emergency_mode: np.ndarray,
        allocation_agent_list: np.ndarray,
        stop_index: np.ndarray,
        status_dim: int,
        emergency_feature_dim: int,
):
    start_index: int = 0
    use_self_greedy = True if len(selected_emergencies) == 0 else False
    # logging.debug("use existing value: {}".format(use_self_greedy))
    # logging.debug("selected emergencies: {}".format(selected_emergencies))
    # can use np.split, does not know performance.
    for i, (stop, emergency_agent_index) in enumerate(zip(stop_index, allocation_agent_list)):
        if emergency_agent_index == -1:
            break
        if use_self_greedy:
            argmin_index = np.argmin(predicted_values[start_index:stop])
        else:
            argmin_index = selected_emergencies[i]
        # logging.debug("argmin index: {}".format(argmin_index))
        if argmin_index != -1:
            offset = int(start_index + argmin_index)
            all_obs[emergency_agent_index] = query_batch[offset]
            my_emergency_mode[emergency_agent_index] = 1
            my_emergency_indices[emergency_agent_index] = actual_emergency_indices[offset]
            my_emergency_target[emergency_agent_index] = query_batch[offset][status_dim:status_dim +
                                                                                        emergency_feature_dim]
            # logging.debug allocation information
            # logging.debug(
            #     f"agent {emergency_agent_index} selected Target:"
            #     f"({my_emergency_target[emergency_agent_index][0]},{my_emergency_target[emergency_agent_index][1]}) "
            #     f"with metric value {predicted_values[offset]}"
            # )
        start_index = stop
    for index in last_round_emergency_mode:
        all_obs[index][status_dim:status_dim + emergency_feature_dim] = my_emergency_target[index]
        # log allocation information
        # logging.debug(
        #     f"agent {index} selected Target:"
        #     f"({my_emergency_target[index][0]},{my_emergency_target[index][1]}) "
        # )
    return all_obs


class CrowdSimAttention(TorchModelV2, nn.Module, BaseMLPMixin):
    emergency_queue: List[deque]

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


def separate_first_task(assignments: Tuple[np.ndarray, np.ndarray]) -> Tuple:
    agents, tasks = assignments
    # Find unique agents and their corresponding indices
    unique_agents, first_task_indices = np.unique(agents, return_index=True)
    # Extract the first task for each unique agent
    first_task_agents = agents[first_task_indices]
    first_task_tasks = tasks[first_task_indices]
    # Find indices of remaining tasks
    remaining_task_indices = np.setdiff1d(np.arange(len(agents)), first_task_indices)
    # Extract remaining tasks
    remaining_task_agents = agents[remaining_task_indices]
    remaining_task_tasks = tasks[remaining_task_indices]
    return (first_task_agents, first_task_tasks), (remaining_task_agents, remaining_task_tasks)
