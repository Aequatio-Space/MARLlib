# MIT License
import logging
import os
import pickle
import pprint
from collections import deque
from functools import reduce
from heapq import heappush, heappop
from typing import Tuple, Union

import numpy as np
import pandas as pd
import wandb
from gym import spaces
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder
from marllib.marl.models.zoo.encoder.triple_encoder import TripleHeadEncoder
from numba import njit
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.typing import Dict, TensorType, List
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds
from torch.distributions import Categorical
from torch import optim

from warp_drive.utils.common import get_project_root
from warp_drive.utils.constants import Constants

rendering_queue_feature = 3

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


class PriorityQueue(list):
    REMOVED: str = '<removed-task>'  # placeholder for a removed task

    def __init__(self, *args, **kwargs):
        super(PriorityQueue, self).__init__(*args, **kwargs)
        self.entry_finder: dict = {}  # mapping of tasks to entries

    def push(self, priority, item):
        heappush(self, (priority, item))
        self.entry_finder[item] = [priority, item]

    def popleft(self):
        while self:
            priority, item = heappop(self)
            if item in self.entry_finder:
                return item
        raise KeyError('pop from an empty priority queue')

    def find(self, task) -> bool:
        if task in self.entry_finder:
            return True
        else:
            return False

    def tolist(self) -> list:
        return [item for _, item in self if item in self.entry_finder]

    def __len__(self):
        return len(self.entry_finder)

    def tasks(self) -> set:
        # note, not ordered.
        return set(self.entry_finder.keys())

    def remove(self, item):
        self.entry_finder.pop(item)

    def removeList(self, items):
        for item in items:
            self.remove(item)


def generate_identity_matrices(n, m):
    # Generate n identity matrices of size m
    identity_matrices = [np.eye(m) for _ in range(n)]
    # Stack the identity matrices horizontally
    result = np.hstack(identity_matrices)
    return result


def split_first_level(d, convert_tensor=True):
    result = {}
    for key, value in d.items():
        parts = key.split('.')
        prefix, actual_key = parts[0], '.'.join(parts[1:])
        if prefix not in result:
            result[prefix] = {}
        if convert_tensor:
            result[prefix][actual_key] = torch.from_numpy(value)
        else:
            result[prefix][actual_key] = value
    return result


def others_target_as_anti_goal(env_agents_pos: np.ndarray):
    # generate distance matrix for all agents in each environment
    num_envs, num_agents = env_agents_pos.shape[0], env_agents_pos.shape[1]
    agents_x = env_agents_pos[..., 0]
    agents_y = env_agents_pos[..., 1]
    env_distances = np.sqrt((agents_x[:, np.newaxis, :] - agents_x[:, :, np.newaxis]) ** 2 +
                            (agents_y[:, np.newaxis, :] - agents_y[:, :, np.newaxis]) ** 2)
    for i in range(num_envs):
        np.fill_diagonal(env_distances[i], 1e6)
    envs_offset = np.repeat(np.arange(0, num_envs * num_agents, num_agents),
                            repeats=num_agents)
    nearest_agent_ids = np.argmin(env_distances, axis=2).ravel() + envs_offset
    flattened_agent_pos = env_agents_pos.reshape(num_envs * num_agents, 2)
    anti_goals_reward = np.linalg.norm(flattened_agent_pos[nearest_agent_ids] - flattened_agent_pos, axis=1)
    clipped_reward = np.clip(anti_goals_reward, a_min=0, a_max=0.3)
    return clipped_reward



@njit
def generate_anti_goal_rewards(anti_goal_reward: np.ndarray,
                               assign_status: np.ndarray,
                               env_agents_pos: np.ndarray,
                               env_id: np.ndarray,
                               emergency_index: np.ndarray):
    # Process each unique environment pair
    unique_envs = np.unique(env_id)
    if len(unique_envs) % 2 and len(unique_envs) > 1:
        # If the number of unique environments is odd, remove the last environment
        unique_envs = unique_envs[:-1]
    # assert len(unique_envs) % 2 == 0, "Number of unique environments must be even"
    for i in range(0, len(unique_envs), 2):
        if i + 1 >= len(unique_envs):  # If there's no pair for the last environment, skip
            break
        env_a, env_b = unique_envs[i], unique_envs[i + 1]
        # Get emergencies for each environment
        emergencies_a = emergency_index[env_id == env_a]
        emergencies_b = emergency_index[env_id == env_b]
        # Find common emergencies
        common_emergencies = np.intersect1d(emergencies_a, emergencies_b)
        agents_env_a = assign_status[env_a, common_emergencies]
        agents_env_b = assign_status[env_b, common_emergencies]
        diff = env_agents_pos[env_a, agents_env_a] - env_agents_pos[env_b, agents_env_b]
        distances = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
        anti_goal_reward[env_a, agents_env_a] = distances
        anti_goal_reward[env_b, agents_env_b] = distances


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
                activation_fn=None),
            # activation_fn=None),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.fc(x)


class AgentSelector(nn.Module):
    def __init__(self, model_config, obs_space, num_agents):
        super(AgentSelector, self).__init__()
        self.num_agents = num_agents
        self.activation = nn.ReLU
        self.encoder = BaseEncoder(model_config, obs_space)
        self.use_2d_state = 'conv_layer' in model_config['custom_model_config']['model_arch_args']
        # self.dropout = nn.Dropout(p=0.3)
        self.out_branch = SlimFC(
            in_size=self.encoder.output_dim,
            out_size=num_agents,
            initializer=normc_initializer(0.01),
            activation_fn=None)

    def forward(self, input_obs, invalid_mask=None, grid=None):
        if self.use_2d_state:
            input_dict = {
                "obs": input_obs,
                "grid": grid
            }
            embedding = self.encoder(input_dict)
        else:
            embedding = self.encoder(input_obs)
        raw_output = self.out_branch(embedding)
        if invalid_mask is not None:
            raw_output -= invalid_mask * (FLOAT_MAX / 2)
        action_scores = torch.nn.functional.softmax(raw_output, dim=1)
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
        self.last_weight_matrix = None
        self.action_space = action_space
        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]
        self.status_dim = self.custom_config["status_dim"]
        self.num_outputs = num_outputs
        self.emergency_feature_dim = self.custom_config["emergency_feature_dim"]
        self.activation = model_config.get("fcnet_activation")
        self.model_arch_args = self.custom_config['model_arch_args']
        self.with_task_allocation = self.model_arch_args['dynamic_zero_shot']
        if self.model_arch_args['use_action_mask']:
            self.custom_config["mask_flag"] = True
        logging.debug(f"CrowdSimNet model_arch_args: {pprint.pformat(self.model_arch_args)}")
        if self.model_arch_args['local_mode']:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selector_type = self.model_arch_args['selector_type']
        self.separate_encoder = (self.model_arch_args["separate_encoder"] or self.custom_config["separate_encoder"])
        self.local_mode = self.model_arch_args['local_mode']
        self.num_envs = self.custom_config["num_envs"] if not self.local_mode else 10
        self.render = self.model_arch_args['render']
        self.render_file_name = self.model_arch_args['render_file_name']
        self.rl_gamma = self.model_arch_args['rl_gamma']
        self.sibling_rivalry = self.model_arch_args['sibling_rivalry']
        self.alpha = self.model_arch_args['alpha']
        self.gen_interval = self.model_arch_args['gen_interval']
        self.emergency_threshold = self.model_arch_args['emergency_threshold']
        self.tolerance = self.model_arch_args['tolerance']
        self.dataset_name = self.model_arch_args['dataset']
        self.look_ahead = self.model_arch_args['look_ahead']
        self.rl_use_cnn = self.model_arch_args['rl_use_cnn']
        self.emergency_queue_length = self.model_arch_args['emergency_queue_length']
        self.intrinsic_mode = self.model_arch_args['intrinsic_mode']
        self.buffer_in_obs = self.model_arch_args['buffer_in_obs']
        self.prioritized_buffer = self.model_arch_args['prioritized_buffer']
        self.NN_buffer = self.model_arch_args['NN_buffer']
        if self.buffer_in_obs:
            self.emergency_dim = self.emergency_queue_length * self.emergency_feature_dim
        else:
            self.emergency_dim = self.emergency_feature_dim
        self.use_aim = self.intrinsic_mode == 'aim'
        if self.use_aim:
            input_dim = self.full_obs_space['obs']['agents_state'].shape[0]
            self.discriminator = Predictor(input_dim).to(self.device)
            self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.custom_config['emergency_dim'] = self.emergency_dim

        if 'checkpoint_path' in self.model_arch_args:
            self.checkpoint_path = self.model_arch_args['checkpoint_path']
        else:
            self.checkpoint_path = None
        self._is_train = False
        self.assignment_sample_batches = []
        self.inputs = None
        self.eval_interval = 1 if self.local_mode else 5
        self.evaluate_count_down = self.eval_interval
        self.trajectory_generated = 0
        self.episode_length = 120
        self.switch_step = -1
        self.step_count = 0
        self.rl_update_interval = max(1, self.num_envs // 10)
        self.reward_mode = self.model_arch_args['reward_mode']
        self.fail_hint = self.model_arch_args['fail_hint']
        self.use_random = self.model_arch_args['use_random']
        self.train_count = 0
        # self.anti_reward_sync_count = 0
        emergency_path_name = os.path.join(get_project_root(), 'datasets',
                                           self.dataset_name, 'emergency_time_loc_1400_1430.csv')
        if os.path.exists(emergency_path_name) and (not self.use_random):
            self.unique_emergencies = pd.read_csv(emergency_path_name)
            self.emergency_count = len(self.unique_emergencies)
        else:
            self.unique_emergencies = None
            self.emergency_count = int(((self.episode_length / self.gen_interval) - 1) * (self.n_agents - 1))
        # self.emergency_label_number = self.emergency_dim // self.emergency_feature_dim + 1
        total_slots = self.n_agents * self.num_envs
        self.emergency_mode = np.zeros(total_slots, dtype=np.bool_)
        self.emergency_indices = np.full(total_slots, -1, dtype=np.int32)
        self.emergency_target = np.full((total_slots, 2), -1, dtype=np.float32)
        if self.prioritized_buffer:
            self.emergency_buffer = [PriorityQueue() for _ in range(total_slots)]
        else:
            self.emergency_buffer = [deque() for _ in range(total_slots)]
        if self.NN_buffer:
            self.weight_generator = Predictor(self.emergency_feature_dim * 2 + 1).to(self.device)
            self.weight_optimizer = torch.optim.Adam(self.weight_generator.parameters(), lr=0.001)
            self.generator_labels = []
            self.generator_inputs = []
        else:
            self.weight_generator = self.generator_labels = self.generator_inputs = None
        # self.anti_goal_reward = np.zeros((self.episode_length, self.num_envs, self.n_agents), dtype=np.float32)
        self.assign_status = np.full((self.num_envs, self.emergency_count), dtype=np.int32, fill_value=-1)
        self.with_programming_optimization = self.model_arch_args['with_programming_optimization']
        self.one_agent_multi_task = self.model_arch_args['one_agent_multi_task']
        self.last_emergency_selection = self.last_emergency_indices = torch.zeros(self.n_agents,
                                                                                  dtype=torch.int32,
                                                                                  device=self.device)
        self.last_rl_transitions = [[] for _ in range(self.num_envs)]
        self.reset_states()
        self.last_emergency_queue_length = self.last_emergency_mode = self.last_emergency_targets = None
        # encoder
        if self.separate_encoder:
            assert self.buffer_in_obs, "buffer must in observation for separate encoder"
            encoder_arch = self.model_arch_args['emergency_encoder']['custom_model_config'][
                'model_arch_args']
            self.encoder_core_arch = encoder_arch['core_arch'] = self.model_arch_args['encoder_core_arch']
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
        self.last_virtual_obs = self.last_buffer_indices = None
        self.last_anti_goal_reward = None
        self.last_predicted_values = None
        self.actors = [self.p_encoder, self.p_branch]
        self.critics = [self.vf_encoder, self.vf_branch]
        self.selector_input_dim = self.n_agents * (2 + self.emergency_queue_length * 2) + 2
        self.actor_initialized_parameters = self.actor_parameters()
        self.aux_selector = None
        agent_selector_arch: dict = model_config['custom_model_config']['agent_selector_arch']

        if not self.rl_use_cnn:
            selector_obs_space = spaces.Dict({
                "obs": spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.selector_input_dim,)),
            })
            agent_selector_arch['custom_model_config']['model_arch_args'].pop('conv_layer')
        else:
            input_shape = (self.n_agents, *self.full_obs_space['obs'][Constants.IMAGE_STATE].shape[1:])
            selector_obs_space = spaces.Dict({
                "obs": spaces.Dict({
                    Constants.VECTOR_STATE: spaces.Box(low=-float('inf'), high=float('inf'),
                                                       shape=(self.selector_input_dim,)),
                    Constants.IMAGE_STATE: spaces.Box(low=-float('inf'), high=float('inf'),
                                                      shape=input_shape),
                })
            })
        for item in ['fcnet_activation', 'conv_activation']:
            agent_selector_arch[item] = model_config[item]
        if len(self.selector_type) == 1:
            mode = self.selector_type[0]
            if mode == "greedy":
                self.selector = GreedySelector(self.n_agents, num_outputs)
            elif mode == 'random':
                self.selector = RandomSelector(self.n_agents, num_outputs)
            elif mode == 'NN':
                input_dim = self.full_obs_space['obs']['agents_state'].shape[0]
                self.selector = Predictor(input_dim).to(self.device)
            elif mode == 'RL':
                self.selector = AgentSelector(agent_selector_arch, selector_obs_space, self.n_agents).to(self.device)
                self.aux_selector = GreedyAgentSelector(self.n_agents * 3 + 2, self.n_agents).to(
                    self.device)
                self.reward_max = -1000
                self.reward_min = 1000
                self.high_level_optim = optim.Adam(self.selector.parameters(), lr=0.001)
            else:
                raise ValueError(f"Unknown selector type {self.selector_type}")
        elif len(self.selector_type) == 2:
            assert 'RL' in self.selector_type or 'NN' in self.selector_type
            if 'greedy' in self.selector_type:
                self.selector = GreedyAgentSelector(self.n_agents * 3 + 2, self.n_agents).to(
                    self.device)
            elif 'random' in self.selector_type:
                self.selector = RandomAgentSelector(self.n_agents * 3 + 2, self.n_agents).to(
                    self.device)
            else:
                raise ValueError(f"Unknown selector type {self.selector_type}")
        # Note, the final activation cannot be tanh, check.
        self.emergency_mode_list = np.zeros([self.episode_length, self.n_agents], dtype=np.bool_)
        self.emergency_target_list = np.zeros([self.episode_length, self.n_agents, 2], dtype=np.float32)
        self.emergency_feature_in_render = 3
        self.emergency_buffer_list = np.full([self.episode_length,
                                              self.n_agents,
                                              self.emergency_queue_length * self.emergency_feature_in_render],
                                             dtype=np.float32, fill_value=-1.0)
        if self.checkpoint_path and not self.render:
            logging.debug(f"Loading checkpoint from {self.checkpoint_path['model_path']}")
            model_state = pickle.load(open(self.checkpoint_path['model_path'], 'rb'))
            worker_state = pickle.loads(model_state['worker'])['state']
            for _, state in worker_state.items():
                executor_weights = state['weights']
                separate_weights = split_first_level(executor_weights)
                for key, value in separate_weights.items():
                    getattr(self, key).load_state_dict(value)
                break
        if wandb.run is not None:
            wandb.watch(models=tuple(self.actors), log='all')

    def reset_states_old(self):
        self.last_emergency_selection = torch.zeros(self.n_agents, device=self.device)
        self.emergency_mode = np.zeros((self.n_agents * self.num_envs), dtype=np.bool_)
        self.emergency_indices = np.full((self.n_agents * self.num_envs), -1, dtype=np.int32)
        self.emergency_target = np.full((self.n_agents * self.num_envs, 2), -1, dtype=np.float32)
        if self.prioritized_buffer:
            self.emergency_buffer = [PriorityQueue() for _ in range(self.n_agents * self.num_envs)]
        else:
            self.emergency_buffer = [deque() for _ in range(self.n_agents * self.num_envs)]

    def reset_states(self):
        self.last_emergency_selection.fill_(0)
        self.emergency_mode.fill(False)
        self.emergency_indices.fill(-1)
        self.emergency_target.fill(-1.0)
        # self.anti_goal_reward.fill(0.0)
        self.assign_status.fill(-1)
        if self.prioritized_buffer:
            self.emergency_buffer = [PriorityQueue() for _ in range(self.n_agents * self.num_envs)]
        else:
            self.emergency_buffer = [deque() for _ in range(self.n_agents * self.num_envs)]
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
            # start_time = time.time()
            self.query_and_assign(flat_inputs, input_dict)
            # logging.debug("--- query_and_assign %s seconds ---" % (time.time() - start_time))
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

    def get_allocation_table(self):
        """
        Get the allocation table for the emergency tasks
        """
        return self.assign_status

    def query_and_assign(self, flat_inputs, input_dict):

        # assert torch.all(self.emergency_queue_lens == torch.tensor([len(q) for q in self.emergency_buffer]))
        if not self._is_train:
            all_obs = flat_inputs[Constants.VECTOR_STATE]
            not_dummy_batch = torch.sum(all_obs) > 0
            numpy_observations = all_obs.cpu().numpy()
            timestep = input_dict['obs']['state'][Constants.VECTOR_STATE][..., -1][0].to(torch.int32).item()
            logging.debug("NN logged timestep: {}".format(timestep))
            if not_dummy_batch:
                env_agents_pos = numpy_observations[:, self.n_agents + 2: self.n_agents + 4].reshape(-1,
                                                                                                     self.n_agents,
                                                                                                     2)
            emergency_status = input_dict['obs']['state'][Constants.VECTOR_STATE][...,
                               self.n_agents * (self.n_agents + 4):-1].reshape(-1, self.emergency_count,
                                                                               5).cpu().numpy()
            emergency_xy = np.stack((emergency_status[0][:, 0],
                                     emergency_status[0][:, 1]), axis=-1)
            target_coverage = emergency_status[..., 3]

            if timestep == 0:
                for item in ['emergency_mode', 'emergency_target']:
                    setattr(self, 'last_' + item, getattr(self, item))
                self.last_emergency_queue_length = [len(q) for q in self.emergency_buffer[:self.n_agents]]
                if self.separate_encoder and self.p_encoder.last_weight_matrix is not None:
                    # for logging purpose
                    self.last_weight_matrix = self.p_encoder.last_weight_matrix.mean(axis=0)
                    self.last_selection = self.p_encoder.last_selection
                # reset network mode
                self.reset_states()
            else:
                # logging.debug(f"Mode before obs: {self.emergency_mode.cpu().numpy()}")
                if not_dummy_batch:
                    if 'oracle' in self.selector_type:
                        # split all_obs into [num_envs] of vectors
                        self.oracle_assignment(all_obs, emergency_xy, target_coverage,
                                               self.emergency_mode, self.emergency_target,
                                               self.emergency_indices)
                    elif 'RL' in self.selector_type:
                        if self.rl_use_cnn:
                            state_info = input_dict['obs']['obs']['grid']
                        else:
                            state_info = None
                        self.rl_assignment(all_obs, env_agents_pos, target_coverage,
                                           self.emergency_buffer, self.n_agents, state_info=state_info,
                                           timestep=timestep, emergency_status=emergency_status)
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
                                self.emergency_buffer,
                                self.emergency_indices,
                                emergency_xy,
                                target_coverage,
                                self.status_dim,
                                self.emergency_feature_dim,
                                self.n_agents,
                                self.emergency_queue_length,
                                self.prioritized_buffer
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
                                        self.assign_status[agent_envs[cur_index]][
                                            actual_emergency_indices[col_ind]] = row_ind
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
                                                my_buffer = self.emergency_buffer[
                                                    allocation_agents[cur_index + agent_index]]
                                                if agent_emergency:
                                                    targets_to_buffer = target_index
                                                else:
                                                    targets_to_buffer = np.delete(target_index,
                                                                                  np.where(target_index == first_id))
                                                for emergency_index in targets_to_buffer:
                                                    if len(my_buffer) < self.emergency_queue_length:
                                                        if self.prioritized_buffer:
                                                            # note, this code is buggy, not implemented with tuple.
                                                            heappush(my_buffer,
                                                                     actual_emergency_indices[emergency_index])
                                                        else:
                                                            my_buffer.append(
                                                                (actual_emergency_indices[emergency_index]))
                                                    else:
                                                        break
                                    logging.debug(f"Assign Status in this batch is {self.assign_status}")
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
            logging.debug(f"Emergency Queue: {self.emergency_buffer[:16]}")
            logging.debug(f"Emergency Queue Length: {[len(q) for q in self.emergency_buffer[:16]]}")
            if self.render or self.evaluate_count_down == 0:
                self.emergency_target_list[timestep] = self.emergency_target[:self.n_agents]
                self.emergency_mode_list[timestep] = self.emergency_mode[:self.n_agents]
                for i, q in enumerate(self.emergency_buffer[:self.n_agents]):
                    # construct a 2d ndarray from queue, the first dim is buffer_length
                    # the second dim is (x,y,index) respectively
                    if self.prioritized_buffer:
                        indices = np.array(q.tolist())
                    else:
                        indices = np.array(list(q))
                    buffer_length = len(indices)
                    if buffer_length > 0:
                        emergencies_in_buffer = emergency_xy[indices]
                        render_result = np.hstack((emergencies_in_buffer, np.array(indices)[:, np.newaxis]))
                        self.emergency_buffer_list[timestep][i, :buffer_length *
                                                                 self.emergency_feature_in_render] = render_result.ravel()
            if timestep == self.episode_length - 1:
                if self.render or self.evaluate_count_down == 0:
                    # WARNING: not modified for numpy.
                    # output all rendering information as csv
                    import pandas as pd
                    from datetime import datetime
                    # concatenate all rendering information
                    final_table = np.concatenate(
                        (self.emergency_mode_list[:, :, np.newaxis], self.emergency_target_list,
                         self.emergency_buffer_list), axis=-1)
                    columns = []
                    for a in range(self.n_agents):
                        columns.extend([f'Agent_{a}_Mode', f'Agent_{a}_Target_X', f'Agent_{a}_Target_Y'] +
                                       [f'Agent_{a}_Queue_{j}_{b}' for b in ['X', 'Y', 'Index']
                                        for j in range(self.emergency_queue_length)])
                    # convert to pandas dataframe
                    all_rendering_info = pd.DataFrame(final_table.reshape(self.episode_length, -1), columns=columns)
                    # save to csv
                    datatime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    all_rendering_info.to_csv(f'{self.render_file_name}_{datatime_str}_{self.trajectory_generated}.csv')
                    logging.debug(f"Detailed Assignment result saved to {self.render_file_name}_{datatime_str}.csv")
                    # reset emergency_buffer_list
                    self.emergency_buffer_list.fill(-1.0)
                if self.evaluate_count_down <= 0:
                    self.evaluate_count_down = self.eval_interval
                    self.trajectory_generated += 1
                else:
                    self.evaluate_count_down -= 1
                    logging.debug(f"Crowdsim Eval Countdown: {self.evaluate_count_down}")
                self.step_count += self.episode_length * self.num_envs
            self.last_virtual_obs = input_dict['obs']['obs']['agents_state']
            self.last_buffer_indices = np.full((self.n_agents * self.num_envs, self.emergency_queue_length),
                                               -1, dtype=np.int32)
            for ind, buffer in enumerate(self.emergency_buffer):
                if self.prioritized_buffer:
                    self.last_buffer_indices[ind, :len(buffer)] = np.array(buffer.tolist())
                else:
                    self.last_buffer_indices[ind, :len(buffer)] = np.array(list(buffer))
            if self.sibling_rivalry:
                if not_dummy_batch:
                    # valid_mask = (self.assign_status != -1) & (target_coverage[::self.n_agents] == 0)
                    # env_id, emergency_index = np.nonzero(valid_mask)
                    # if len(env_id) > 0:
                    new_anti_goal_reward = others_target_as_anti_goal(
                        env_agents_pos,
                    )
                    self.last_anti_goal_reward = new_anti_goal_reward
                    # else:
                    #     self.last_anti_goal_reward = np.zeros(self.num_envs * self.n_agents)
                    # self.anti_goal_reward[timestep] = new_anti_goal_reward
                else:
                    self.last_anti_goal_reward = np.zeros(all_obs.shape[0])
        else:
            try:
                input_dict['obs']['obs']['agents_state'] = input_dict['virtual_obs']
            except KeyError:
                logging.debug("No virtual obs found")
        if self.custom_config['mask_flag']:
            batch_size = input_dict['obs']['obs']['agents_state'].shape[0]
            # warning, this may not adapt to MultiDiscrete and Continuous
            mask = torch.zeros((batch_size, self.num_outputs), dtype=torch.float32)
            agents_pos = input_dict['obs']['obs']['agents_state'][:, self.n_agents + 2: self.n_agents + 4]
            emergencies_pos = input_dict['obs']['obs']['agents_state'][:, self.status_dim:self.status_dim +
                                                                                          self.emergency_feature_dim]
            no_emergencies_mask = torch.all(emergencies_pos == 0, dim=-1)
            mask[no_emergencies_mask, :] = 1
            if self.emergency_feature_dim > 2:
                emergencies_pos = emergencies_pos[..., :2]
            difference = emergencies_pos - agents_pos
            # Compute the signs of the elements in some_array
            quadrant_x = torch.sign(difference[:, 0]).to(torch.int32)
            quadrant_y = torch.sign(difference[:, 1]).to(torch.int32)
            quadrant_labels = torch.zeros(batch_size, dtype=torch.int32)
            quadrant_labels[(quadrant_x == 1) & (quadrant_y == 1)] = 1
            quadrant_labels[(quadrant_x == -1) & (quadrant_y == 1)] = 2
            quadrant_labels[(quadrant_x == -1) & (quadrant_y == -1)] = 3
            quadrant_labels[(quadrant_x == 1) & (quadrant_y == -1)] = 4
            for index, item in enumerate(quadrant_labels):
                mask[index, Constants.ACTION_VALID_DICT[item.item()]] = 1
            input_dict['obs']['action_mask'] = mask.to(self.device)

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
            #              len(self.emergency_buffer[j + cur_index]) +
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
                    self.emergency_buffer[actual_agents[agent_id]].append(
                        actual_emergency_indices[emergency_id])
            else:
                logging.debug("Failed to solve the optimization problem, "
                              "No allocation is made")

    def rl_assignment(self, all_obs, env_agents_pos, target_coverage, my_emergency_buffer, n_agents,
                      state_info=None, timestep=None, emergency_status=None):
        all_env_obs = all_obs.clone().detach().reshape(-1, n_agents, self.status_dim + self.emergency_dim)
        env_target_coverage = target_coverage[::n_agents]
        emergency_xy = np.stack([emergency_status[..., 0], emergency_status[..., 1]], axis=-1)
        self.clear_buffer(env_target_coverage, my_emergency_buffer, emergency_xy,
                          env_agents_pos, n_agents, emergency_status)
        self.do_rl_assignment(all_env_obs, emergency_xy, env_target_coverage, my_emergency_buffer,
                              n_agents, state_info, timestep, emergency_status)
        self.output_rl_assignment(all_obs, emergency_xy, my_emergency_buffer)

    def clear_buffer(self, env_target_coverage, my_emergency_buffer, emergency_xy,
                     env_agents_pos, n_agents, emergency_status=None):
        env_covered_emergency = env_target_coverage == 1
        for i in range(self.num_envs):
            offset = i * n_agents
            for j in range(n_agents):
                actual_agent_id = offset + j
                if self.prioritized_buffer:
                    my_buffer: PriorityQueue = my_emergency_buffer[actual_agent_id]
                    covered_emergency = set(np.nonzero(env_covered_emergency[i])[0])
                    new_buffer = PriorityQueue()
                    for item in my_buffer.tolist():
                        if item not in covered_emergency:
                            distance = np.linalg.norm(env_agents_pos[i][j] - emergency_xy[i][item])
                            aoi = emergency_status[i][item][2]
                            if self.NN_buffer:
                                new_priority = distance * (1 - aoi / self.emergency_threshold)
                                # current_emergency_status = np.concatenate(
                                #     [env_agents_pos[i][j], emergency_status[i][item][:3]])
                                # input = torch.from_numpy(current_emergency_status).unsqueeze(0).to(self.device)
                                # new_priority = self.weight_generator(input).item()
                                # self.generator_labels.append(
                                #     torch.tensor(distance * (1 - aoi / self.emergency_threshold)))
                                # self.generator_inputs.append(input.squeeze(0))
                            else:
                                new_priority = distance
                            new_buffer.push(new_priority, item)
                        # else:
                        #     self.assign_status[i][item] = -1
                    my_emergency_buffer[actual_agent_id] = new_buffer
                else:
                    my_buffer: deque = my_emergency_buffer[actual_agent_id]
                    if len(my_buffer) == 0:
                        continue
                    head_emergency = my_buffer[0]
                    while len(my_buffer) > 0 and env_covered_emergency[i][head_emergency]:
                        # logging.debug(f"Emergency {my_buffer[0]} is covered by agent {actual_agent_id}")
                        my_buffer.popleft()

    def do_rl_assignment(self, all_env_obs, emergency_xy, env_target_coverage,
                         my_emergency_buffer, n_agents, state_info=None, timestep=None,
                         emergency_status=None):
        """
        Make emergency assignment for each agent in each environment.
        """
        env_valid_emergencies = (env_target_coverage == 0) & (self.assign_status == -1)
        env_agents_pos = all_env_obs[:, :, n_agents + 2: n_agents + 4]
        env_buffer_lens = torch.tensor([len(q) for q in my_emergency_buffer]).reshape(
            self.num_envs, self.n_agents).to(self.device)
        invalid_mask: torch.Tensor = env_buffer_lens >= self.emergency_queue_length
        all_buffer_xy = torch.zeros((self.num_envs * self.n_agents, self.emergency_queue_length * 2),
                                    device=self.device)
        non_empty_buffers = torch.nonzero(env_buffer_lens.flatten() > 0)
        if state_info is not None:
            state_info = state_info.reshape(-1, self.n_agents, *state_info.shape[2:])
        for i in non_empty_buffers:
            if self.prioritized_buffer:
                indices = my_emergency_buffer[i].tolist()
            else:
                indices = list(my_emergency_buffer[i])
            all_buffer_xy[i, :env_buffer_lens[i // self.n_agents, i % self.n_agents] * 2] = \
                torch.from_numpy(emergency_xy[i][indices].flatten()).to(self.device)
        if len(env_valid_emergencies) == 0:
            return
        for i in range(self.num_envs):
            valid_emergencies = env_valid_emergencies[i]
            emergencies = emergency_xy[i][valid_emergencies]
            logging.debug(f"Valid Emergencies:\n{emergencies}")
            received_tasks = len(emergencies)
            if received_tasks > 0:
                single_invalid_mask = invalid_mask[i]
                if torch.all(single_invalid_mask):
                    logging.debug("All agents are full, no assignment is made")
                    continue
                offset = i * n_agents
                agents_pos = env_agents_pos[i]
                if self.look_ahead:
                    for j in range(self.n_agents):
                        this_buffer = my_emergency_buffer[offset + j]
                        if len(this_buffer) > 0:
                            if self.prioritized_buffer:  # implementation may not be correct
                                _, last_dest = this_buffer[-1]
                            else:
                                last_dest = this_buffer[-1]
                            agents_pos[j] = torch.from_numpy(emergency_xy[i][last_dest])
                this_assign_status = self.assign_status[i]
                agents_buffer_lens = env_buffer_lens[i].unsqueeze(-1)
                actual_emergency_indices = np.nonzero(valid_emergencies)[0]
                obs_list, action_list, invalid_mask_list = [], [], []
                state_grid_list = []
                for k, emergency in enumerate(emergencies):
                    # relative_pos = agents_pos - np.tile(emergency, reps=self.n_agents).reshape(self.n_agents, 2)
                    if torch.all(single_invalid_mask):
                        logging.debug("All agents are full, no further assignment will be made")
                        break
                    torch_emergency = torch.from_numpy(emergency).to(self.device)
                    if len(self.selector_type) == 1 and self.selector_type[0] == 'RL':
                        selector_obs = torch.cat([agents_pos,
                                                  all_buffer_xy[offset:offset + n_agents]], dim=1).flatten()
                        final_obs = torch.cat([selector_obs, torch_emergency]).unsqueeze(0)
                    else:
                        selector_obs = torch.cat([agents_pos, agents_buffer_lens], dim=1).flatten()
                        final_obs = torch.cat([selector_obs, torch_emergency])
                        final_obs = final_obs.unsqueeze(0)
                    obs_list.append(final_obs.cpu().numpy())
                    if self.step_count > self.switch_step:
                        logging.debug("Using RL Agent Selector")
                        if self.rl_use_cnn:
                            assert state_info is not None
                            this_env_state_grid = state_info[i].unsqueeze(0)
                            state_grid_list.append(this_env_state_grid)
                            probs = self.selector(final_obs, single_invalid_mask,
                                                  grid=this_env_state_grid)
                        else:
                            probs = self.selector(final_obs, single_invalid_mask)
                    else:
                        logging.debug("Using Greedy Agent Selector")
                        probs = self.aux_selector(final_obs, single_invalid_mask)
                    logging.debug("Probs: {}".format(probs))
                    action_dists: Categorical = Categorical(probs=probs)
                    action = int(action_dists.sample().cpu().numpy())
                    invalid_mask_list.append(single_invalid_mask.cpu().numpy())
                    # log_prob = action_dists.log_prob(action)
                    # action_logp_list.append(log_prob.cpu().numpy())
                    action_list.append(action)
                    # reward = -np.linalg.norm(agents_pos[actions].cpu().numpy() - emergencies, axis=1)
                    agent_id = offset + action
                    current_buffer = my_emergency_buffer[agent_id]
                    if len(current_buffer) < self.emergency_queue_length:
                        this_index = actual_emergency_indices[k]
                        if self.prioritized_buffer:
                            # calculate the distance between selected agent and emergency
                            distance = np.linalg.norm(agents_pos[action].cpu().numpy() - emergency)
                            aoi = emergency_status[i][this_index][2]
                            if self.NN_buffer:
                                new_priority = distance * (1 - aoi / self.emergency_threshold)
                                current_buffer.push(new_priority, this_index)
                                # current_emergency_status = np.concatenate(
                                #     [agents_pos[action].cpu().numpy(), emergency_status[i][actual_emergency_indices[k]][:3]])
                                # input = torch.from_numpy(current_emergency_status).unsqueeze(0).to(self.device)
                                # new_priority = self.weight_generator(input).item()
                                # self.generator_labels.append(
                                #     torch.tensor(distance * (1 - aoi / self.emergency_threshold)))
                                # self.generator_inputs.append(input.squeeze(0))
                            else:
                                current_buffer.push(distance, this_index)
                        else:
                            current_buffer.append(this_index)
                        agents_buffer_lens[action] += 1
                        all_buffer_xy[agent_id][2 * (agents_buffer_lens[action] - 1):
                                                2 * agents_buffer_lens[action]] = torch_emergency
                        single_invalid_mask[action] = agents_buffer_lens[action] >= self.emergency_queue_length
                        if self.look_ahead:
                            agents_pos[action] = torch_emergency
                        this_assign_status[this_index] = action
                        logging.debug(f"Assign Status in Env {i} is {this_assign_status}")
                # save the obs, action and reward for assignment agent.
                if len(obs_list) > 0:
                    construct_dict = {
                        SampleBatch.OBS: np.vstack(obs_list),
                        SampleBatch.ACTIONS: np.array(action_list, dtype=np.int32),
                        'invalid_mask': np.array(invalid_mask_list, dtype=np.bool_),
                        SampleBatch.REWARDS: np.full_like(action_list, -1, dtype=np.float32),
                        'timesteps': np.full_like(action_list, timestep, dtype=np.int32)
                    }
                    if self.rl_use_cnn:
                        construct_dict[Constants.IMAGE_STATE] = torch.cat(state_grid_list, dim=0).cpu().numpy()
                    self.last_rl_transitions[i].append(SampleBatch(construct_dict))

    def output_rl_assignment(self, all_obs, emergency_xy, all_buffer):
        for i, buffer in enumerate(all_buffer):
            my_len = len(buffer)
            if my_len > 0:
                if self.prioritized_buffer:
                    head_distance, buffer_head = buffer[0]
                else:
                    buffer_head = buffer[0]
                    head_distance = None
                new_emergency_xy = emergency_xy[i // self.n_agents][buffer_head]
                self.emergency_indices[i] = buffer_head
                self.emergency_target[i] = new_emergency_xy
                self.emergency_mode[i] = 1
                agent_pos = all_obs[i][self.n_agents + 2: self.n_agents + 4]
                if self.buffer_in_obs:
                    buffer_as_obs = torch.zeros(self.emergency_dim, device=self.device)
                    if self.prioritized_buffer:
                        indices = buffer.tolist()
                    else:
                        indices = list(buffer)
                    this_emergency_xy = torch.from_numpy(emergency_xy[i // self.n_agents][indices]).to(self.device)
                    if self.emergency_feature_dim > 2:
                        distances = torch.norm(agent_pos - this_emergency_xy, dim=1)
                        buffer_features = torch.cat([this_emergency_xy, distances.unsqueeze(-1)], dim=1).flatten()
                    else:
                        buffer_features = this_emergency_xy.flatten()
                    buffer_as_obs[:my_len * self.emergency_feature_dim] = buffer_features
                    all_obs[i][self.status_dim:self.status_dim + self.emergency_dim] = buffer_as_obs
                else:
                    if not self.prioritized_buffer:
                        head_distance = np.linalg.norm(agent_pos - new_emergency_xy)
                    if self.emergency_feature_dim > 2:
                        new_emergency_xy = np.concatenate([new_emergency_xy, [head_distance]])
                    all_obs[i][self.status_dim:self.status_dim + self.emergency_feature_dim] = \
                        torch.from_numpy(new_emergency_xy)
            else:
                self.emergency_indices[i] = -1
                self.emergency_target[i].fill(-1)
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
        my_emergency_buffer: Union[list[deque], list[PriorityQueue]],
        my_emergency_index: np.ndarray,
        emergency_xy: np.ndarray,
        target_coverage: np.ndarray,
        status_dim: int,
        emergency_feature_dim: int,
        n_agents: int,
        max_queue_length: int,
        prioritized: bool
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
        my_buffer: deque = my_emergency_buffer[i]
        #     # logging.debug(f"index: {i}")
        if my_emergency_mode[i] and this_coverage[my_emergency_index[i]]:
            # target is covered, reset emergency mode
            # logging.debug("my_emergency_index: {}".format(my_emergency_index[i]))
            if len(my_buffer) == 0:
                my_emergency_mode[i] = 0
                my_emergency_target[i] = -1
                my_emergency_index[i] = -1
            else:
                new_emergency = my_buffer.popleft()
                while this_coverage[new_emergency] and len(my_buffer) > 0:
                    new_emergency = my_buffer.popleft()
                my_emergency_index[i] = new_emergency
                logging.debug(f"Selecting from buffer: {my_emergency_index[i]}")
                my_emergency_target[i] = emergency_xy[my_emergency_index[i]]
                logging.debug(f"Emergency target ({my_emergency_target[i][0]},{my_emergency_target[i][1]}) "
                              f"is allocated to agent {i}")
    last_round_emergency = my_emergency_mode & last_round_emergency

    for i, this_coverage in enumerate(target_coverage):
        if not my_emergency_mode[i]:
            env_num = i // n_agents
            offset = env_num * n_agents
            valid_emergencies = this_coverage == 0
            # convert all queue entries in an env into a list
            env_buffers = my_emergency_buffer[offset:offset + n_agents]
            # unwrap list of list, remove emergencies in agents buffer.
            additional_emergencies = [item for sublist in env_buffers for item in sublist]
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
