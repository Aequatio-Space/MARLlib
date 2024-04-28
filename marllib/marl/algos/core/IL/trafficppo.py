"""
PyTorch policy class used for PPO.
"""
import datetime
import logging
import os
import pickle
import random
import re
from copy import deepcopy
from typing import Dict, List, Type, Union, Optional
from math import prod
import gym
import numpy as np
from marllib.marl.algos.wandb_trainers import WandbPPOTrainer
from numba import njit
from ray.rllib.agents.ppo import PPOTorchPolicy, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import (kl_and_loss_stats, ppo_surrogate_loss,
                                                   compute_gae_for_sample_batch, vf_preds_fetches)
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict, AgentID
from torch import optim
from torch.distributions import Categorical
from tqdm import tqdm

from envs.crowd_sim.utils import get_emergency_labels
from warp_drive.utils.constants import Constants

INTRINSIC_REWARDS = 'intrinsic_rewards'

RAW_ASSIGN_REWARDS = 'assign_rewards'

SURVEILLANCE_REWARDS = 'surveillance_rewards'

ANTI_GOAL_REWARD = 'anti_goal_reward'

ORIGINAL_REWARDS = 'original_rewards'

EMERGENCY_REWARD_INCREMENT = 10.0

emergency_feature_in_state = 5

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

VIRTUAL_OBS = 'virtual_obs'

use_large_emergency = False

# dirty hack should be removed later
episode_length = 120


def mean_length(list_of_tensors: list[torch.Tensor]) -> torch.Tensor:
    lengths = [len(tensor) for tensor in list_of_tensors]
    if len(lengths) > 0:
        return torch.tensor(lengths).float().mean()
    else:
        return torch.tensor(0.0)


def find_ascending_sequences(arr_tensor: torch.Tensor, min_length=5) -> list[torch.Tensor]:
    """
    find the ascending sequences in a tensor, return a list of ascending sequences.
    """
    # Find the differences between consecutive elements
    diffs = torch.diff(arr_tensor)
    # Find the indices where the sequence starts
    fragments = []
    length_of_diff = len(diffs)
    i = 0
    while i < length_of_diff:
        if diffs[i] > 0:
            start_index = i
            length = 1
            while i < length_of_diff and diffs[i] > 0:
                i += 1
                length += 1
            if length > min_length:
                fragments.append(torch.arange(start_index, i))
        i += 1
    return fragments


def generate_ascending_sequence_vectorized(count, limit):
    # increments = np.cumsum(np.random.randint(1, int(limit / count), (count,), dtype=np.int32))
    # return increments
    import math
    segment_length = math.ceil(limit / count)
    base = np.arange(0, limit, segment_length)
    offsets = np.random.randint(0, segment_length, (count,), dtype=np.int32)
    sequence = base + offsets
    return np.where(sequence < limit, sequence, limit - 1)


def relabel_for_sample_batch(
        policy: TorchPolicy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    """
    relabel expected goal for agents
    """
    k = 0
    goal_number = 5

    selector_type = policy.model.selector_type
    intrinsic_mode = policy.model.intrinsic_mode
    intrinsic_mode = intrinsic_mode
    use_intrinsic = intrinsic_mode != 'none'
    use_gdan = policy.model.use_gdan or policy.model.use_gdan_lstm
    use_distance_intrinsic = intrinsic_mode != 'aim' or policy.model.step_count < policy.model.switch_step
    # logging.debug("use_distance_intrinsic: %s", use_distance_intrinsic)
    # logging.debug("step_count: %d, switch_step: %d", policy.model.step_count, policy.model.switch_step)
    status_dim = policy.model.status_dim
    emergency_dim = policy.model.emergency_dim
    emergency_feature_dim = policy.model.emergency_feature_dim
    use_action_label = policy.model.use_action_label
    # postprocess of rewards
    num_agents = policy.model.n_agents
    # postprocess extra_batches
    try:
        virtual_obs = sample_batch[VIRTUAL_OBS]
        sample_batch[SampleBatch.OBS][..., :status_dim + emergency_dim] = virtual_obs
    except KeyError:
        pass
    if policy.model.with_task_allocation:
        this_emergency_count = policy.model.emergency_count
        observation = sample_batch[SampleBatch.OBS]
        emergency_states = get_emergency_state(emergency_dim, sample_batch, num_agents, status_dim,
                                               this_emergency_count)
        agents_position = observation[..., num_agents + 2: num_agents + 4]
        num_envs = policy.model.num_envs
        batch_size = emergency_states.shape[0]
        if policy.model.buffer_in_obs:
            if policy.model.separate_encoder and 'gumbel' in policy.model.encoder_core_arch:
                emergency_position = sample_batch['executor_obs'][...,
                                     status_dim:status_dim + emergency_feature_dim]
            else:
                emergency_position = observation[..., status_dim:status_dim + emergency_dim].reshape(
                    batch_size, -1, emergency_feature_dim
                )
        else:
            emergency_position = observation[..., status_dim:status_dim + emergency_dim]
        if emergency_feature_dim > 2:
            emergency_position = emergency_position[..., :2]
        if use_distance_intrinsic:
            if policy.model.sibling_rivalry:
                assert ANTI_GOAL_REWARD in sample_batch, ANTI_GOAL_REWARD + " is not in sample_batch."
                anti_goal_distances = sample_batch[ANTI_GOAL_REWARD]
            else:
                anti_goal_distances = None
            # rearrange emergency according to weight matrix
            indices = None
            if policy.model.separate_encoder:
                core_arch = policy.model.encoder_core_arch
                if 'attention' in core_arch:
                    if 'gumbel' not in core_arch:
                        if 'weight_matrix' not in sample_batch:
                            raise ValueError("weight_matrix is not in sample_batch.")
                        rearranged_indices = sample_batch['weight_matrix'].argsort()
                        for i in range(batch_size):
                            emergency_position[i] = emergency_position[i][rearranged_indices[i]]
                    else:
                        indices = sample_batch['buffer_indices'][np.arange(batch_size), sample_batch['selection']]

            intrinsic = calculate_intrinsic(agents_position, emergency_position, emergency_states,
                                            emergency_threshold=policy.model.emergency_threshold,
                                            anti_goal_distances=anti_goal_distances, indices=indices,
                                            alpha=policy.model.alpha, mode=intrinsic_mode)
        else:
            discriminator: nn.Module = policy.model.discriminator
            cur_obs = sample_batch[SampleBatch.CUR_OBS][..., :status_dim + emergency_dim]
            next_obs = sample_batch[SampleBatch.NEXT_OBS][..., :status_dim + emergency_dim]
            cur_valid_mask = cur_obs[..., status_dim:status_dim + emergency_dim].sum(axis=-1) != 0
            next_valid_mask = next_obs[..., status_dim:status_dim + emergency_dim].sum(axis=-1) != 0
            valid_mask = np.logical_and(cur_valid_mask, next_valid_mask)
            next_obs_potential = discriminator(torch.from_numpy(next_obs).to(policy.model.device))
            delta = (policy.model.reward_max - policy.model.reward_min)
            intrinsic = (next_obs_potential.cpu().numpy() - policy.model.reward_max) / delta
            intrinsic = intrinsic.ravel() * valid_mask
        # intrinsic[torch.mean(distance_between_agents < 0.1) > 0] *= 1.5
        sample_batch[ORIGINAL_REWARDS] = deepcopy(sample_batch[SampleBatch.REWARDS])
        sample_batch[INTRINSIC_REWARDS] = intrinsic
        if isinstance(sample_batch[SampleBatch.REWARDS], torch.Tensor):
            sample_batch[SampleBatch.REWARDS] += torch.from_numpy(intrinsic)
        else:
            sample_batch[SampleBatch.REWARDS] += intrinsic
        if use_gdan:
            vec_dim = policy.model.full_obs_space['obs']['agents_state'].shape[0]
            grid_dim = policy.model.full_obs_space['obs']['grid'].shape
            if intrinsic_mode == 'none':
                raw_rewards = sample_batch[SampleBatch.REWARDS]
            else:
                raw_rewards = sample_batch[ORIGINAL_REWARDS]
            if use_action_label:
                from envs.crowd_sim.utils import generate_quadrant_labels
                eps = 0.3
                emergencies_mask = ~np.all(emergency_position == 0, axis=-1)
                # randomly flip 0 in emergencies_mask to 1
                random_values = np.random.rand(*emergencies_mask.shape)
                # Identify where arr is False AND the random value is less than eps
                # These are the positions where we'll flip False to True
                flip_positions = (emergencies_mask == False) & (random_values < eps)
                # Flip the identified False positions to True
                emergencies_mask[flip_positions] = True
                matched_obs = virtual_obs[emergencies_mask]
                agents_pos = matched_obs[..., num_agents + 2: num_agents + 4]
                new_emergency_position = emergency_position[emergencies_mask]
                aoi_grids = sample_batch[SampleBatch.OBS][emergencies_mask,
                            vec_dim: vec_dim + prod(grid_dim)].reshape(-1, *grid_dim)
                list_of_dict = []
                for i in range(len(aoi_grids)):
                    list_of_dict.append({
                        'agents_state': matched_obs[i],
                        'grid': aoi_grids[i]
                    })
                labels = generate_quadrant_labels(agents_pos, new_emergency_position,
                                                  new_emergency_position.shape[0]).cpu().numpy()
                policy.model.goal_storage.putAll(list_of_dict, labels)
            else:
                emergency_handled_mask = raw_rewards >= 0.99
                if np.any(emergency_handled_mask):
                    matched_obs = virtual_obs[emergency_handled_mask]
                    aoi_grids = sample_batch[SampleBatch.OBS][emergency_handled_mask,
                                vec_dim: vec_dim + prod(grid_dim)].reshape(-1, *grid_dim)
                    list_of_dict = []
                    for i in range(len(aoi_grids)):
                        list_of_dict.append({
                            'agents_state': matched_obs[i],
                            'grid': aoi_grids[i]
                        })
                    labels = get_emergency_labels(matched_obs, status_dim)
                    policy.model.goal_storage.putAll(list_of_dict, labels)
        if policy.model.NN_buffer:
            last_emergency_state = emergency_states[-1]
            handled_emergencies = last_emergency_state[..., -1] == sample_batch['agent_index'][0]
            if np.any(handled_emergencies):
                valid_steps = (emergency_states[:, handled_emergencies, 3] == 0).T
                valid_emergencies_indices = np.nonzero(handled_emergencies)[0]
                observation_list = []
                label_list = []
                for single_valid_step, index in zip(valid_steps, valid_emergencies_indices):
                    generator_obs = np.hstack([observation[single_valid_step, num_agents + 2:num_agents + 4],
                                               emergency_states[single_valid_step, index, :3]])
                    observation_list.append(generator_obs)
                    # labels are descending number normalized by episode length, the largest number is np.count_nonzero(single_valid_step)
                    longest_timestep = np.count_nonzero(single_valid_step)
                    labels = np.arange(start=longest_timestep, stop=0, step=-1) / episode_length
                    label_list.append(labels)
                # concatenate the observation list
                observation_list = np.vstack(observation_list)
                label_list = np.concatenate(label_list)
                policy.model.generator_inputs.append(observation_list)
                policy.model.generator_labels.append(label_list)


    return label_done_masks_and_calculate_gae_for_sample_batch(policy, sample_batch,
                                                               other_agent_batches, episode)

    # try, send relabeled trajectory only.


def NN_bootstrap_reward(emergency_dim, emergency_position, emergency_states, observation, policy, selector_type,
                        status_dim):
    assert 'NN' == selector_type[0] and len(
        selector_type) == 1, "only NN selector supports bootstrapping reward."
    inputs = torch.from_numpy(observation[..., :status_dim + emergency_dim]).float()
    agent_metric_estimations = policy.model.selector(inputs.to(policy.model.device))
    if len(agent_metric_estimations) > 0:
        agent_metric_estimations = agent_metric_estimations.squeeze(-1)
    age_of_informations = emergency_states[..., 2]
    mask = emergency_position.sum(axis=-1) != 0
    all_emergencies_position = np.stack([emergency_states[..., 0], emergency_states[..., 1]], axis=-1)
    indices = match_aoi(all_emergencies_position, emergency_position, mask)
    intrinsic = -agent_metric_estimations.cpu().numpy()
    intrinsic *= (mask * age_of_informations[np.arange(len(age_of_informations)), indices])
    return intrinsic


def get_start_indices_of_segments(array):
    array = np.array(array)
    unique_pairs, inverse_indices = np.unique(array, axis=0, return_inverse=True)
    # Get the start indices of each segment
    start_indices = sorted([np.where(inverse_indices == i)[0][0] for i in range(len(unique_pairs))])
    return start_indices


def label_done_masks_and_calculate_gae_for_sample_batch(
        policy: TorchPolicy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    status_dim = policy.model.status_dim
    emergency_dim = policy.model.emergency_feature_dim
    if policy.model.intrinsic_mode == 'NN':
        # check whether numbers between status_dim:status_dim + emergency_dim are all zeros, one row per value
        emergency_obs = sample_batch[SampleBatch.OBS][..., status_dim:status_dim + emergency_dim]
        no_emergency_mask = np.nansum(np.abs(emergency_obs), axis=1) == 0
        emergencies_masks = []
        separate_batches = []
        last_index = 0
        try:
            raw_rewards = sample_batch[ORIGINAL_REWARDS]
        except KeyError:
            raw_rewards = sample_batch[SampleBatch.REWARDS]
        separate_result = get_start_indices_of_segments(emergency_obs)
        for index in separate_result[1:]:
            separate_batches.append(sample_batch[last_index:index + 1].copy())
            emergencies_masks.append(no_emergency_mask[last_index:index + 1])
            last_index = index + 1
        if len(separate_batches) == 0:
            sample_batch['labels'] = np.where(no_emergency_mask, 1, -1)
            return compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)
        if last_index < len(raw_rewards):
            separate_batches.append(sample_batch[last_index:].copy())
            emergencies_masks.append(no_emergency_mask[last_index:])
        labeled_batches = []
        length = len(separate_batches)
        for i, (mask, batch) in enumerate(zip(emergencies_masks, separate_batches)):
            batch_length = len(batch[SampleBatch.REWARDS])
            if i != length - 1:
                labels = np.arange(start=batch_length, stop=0, step=-1) / episode_length
                batch['labels'] = np.where(mask, labels, -1)
            else:
                batch['labels'] = np.full(batch_length,
                                          -1)  # discard last batch by default since it contains truncated result.
            labeled_batches.append(batch)
            # labeled_batches.append(compute_gae_for_sample_batch(policy, batch, other_agent_batches, episode))
        full_batch = SampleBatch.concat_samples(labeled_batches)
    else:
        full_batch = sample_batch
    return compute_gae_for_sample_batch(policy, full_batch, other_agent_batches, episode)


def fix_interval_relabeling(extra_batches, goal_number, num_agents, sample_batch, policy: TorchPolicy):
    status_dim = policy.model.status_dim
    emergency_features = policy.model.emergency_feature_dim
    emergency_count = policy.model.emergency_count
    if use_large_emergency:
        emergency_dim = emergency_count * emergency_features
    else:
        emergency_dim = emergency_features
    for i in range(len(extra_batches)):
        original_batch = extra_batches[i]
        original_obs = original_batch[SampleBatch.OBS]
        obs_shape = original_batch[SampleBatch.OBS].shape
        bins = np.linspace(0, obs_shape[0], emergency_count)
        agents_position = sample_batch[SampleBatch.OBS][goal_number::goal_number, num_agents + 2: num_agents + 2 + 2]
        # the agent_position in the same step length should be the same
        for j in range(obs_shape[0] - goal_number):
            if use_large_emergency:
                emergency_location = int((np.digitize(np.array([j]), bins) - 1) * emergency_features)
            else:
                emergency_location = 0
            original_obs[j][status_dim:status_dim + emergency_dim] = 0
            my_fix_goal = agents_position[j // goal_number]
            original_obs[j][status_dim + emergency_location:
                            status_dim + emergency_location + emergency_features] = my_fix_goal
        original_batch[SampleBatch.REWARDS][np.arange(goal_number - 1, obs_shape[0],
                                                      goal_number, dtype=np.int32)] += EMERGENCY_REWARD_INCREMENT
        extra_batches[i] = original_batch


def mapping_relabeling(extra_batches, start_timestep, num_agents, sample_batch, policy: TorchPolicy):
    status_dim = policy.model.status_dim
    emergency_features = emergency_dim = policy.model.emergency_feature_dim
    for i in range(len(extra_batches)):
        original_batch = extra_batches[i]
        # obs_shape = original_batch[SampleBatch.OBS].shape
        agents_position = sample_batch[SampleBatch.OBS][..., num_agents + 2: num_agents + 2 + 2]
        goals_to_be_filled = agents_position[start_timestep:, :]
        # sample an ascending sequence of with length goal_number
        original_obs = original_batch[SampleBatch.OBS]
        original_obs[:len(original_obs) - start_timestep, status_dim:status_dim + emergency_dim] = 0
        original_obs[:len(original_obs) - start_timestep,
        status_dim: status_dim + emergency_features] = goals_to_be_filled
        extra_batches[i] = original_batch


def future_multi_goal_relabeling(extra_batches, goal_number, num_agents, sample_batch, policy: TorchPolicy):
    status_dim = policy.model.status_dim
    emergency_feature_dim = policy.model.emergency_feature_dim
    emergency_count = policy.model.emergency_count
    if use_large_emergency:
        emergency_features = emergency_count * emergency_feature_dim
    else:
        emergency_features = emergency_feature_dim
    # try relabeling at the first column only?
    for i in range(len(extra_batches)):
        original_batch = extra_batches[i]
        obs_shape = original_batch[SampleBatch.OBS].shape
        agents_position = sample_batch[SampleBatch.OBS][..., num_agents + 2: num_agents + 2 + 2]
        # sample an ascending sequence of with length goal_number
        sequence = generate_ascending_sequence_vectorized(goal_number, obs_shape[0])
        logging.debug(f"sequence: {sequence}")
        original_obs = original_batch[SampleBatch.OBS]
        j = 0
        fill_index = 0
        # values from 0 to 16 ((emergnecy_count - 1) * feature_num) with step 2
        # bin from 0 to episode_length with the same length of values
        bins = np.linspace(0, obs_shape[0], emergency_count)
        sequence_length = len(sequence)
        while fill_index < sequence_length and j < obs_shape[0]:
            if use_large_emergency:
                emergency_location = int((np.digitize(np.array([j]), bins) - 1) * emergency_features)
            else:
                emergency_location = 0
            logging.debug(f"emergency_location: {emergency_location}")
            goal_to_fill = agents_position[sequence[fill_index]]
            while j < obs_shape[0]:
                original_obs[j][status_dim:status_dim + emergency_features] = 0
                original_obs[j][status_dim + emergency_location:
                                status_dim + emergency_location + emergency_features] = goal_to_fill
                if np.linalg.norm(goal_to_fill - agents_position[j]) < 0.05:
                    original_batch[SampleBatch.REWARDS][j] += 10
                    j += 1
                    break
                j += 1
            fill_index += 1
        extra_batches[i] = original_batch


def compute_gae_and_intrinsic_for_sample_batch(
        policy: TorchPolicy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    """GAE (generalized advantage estimations) to a trajectory.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy (Policy): The Policy used to generate the trajectory
            (`sample_batch`)
        sample_batch (SampleBatch): The SampleBatch to postprocess.
        other_agent_batches (Optional[Dict[PolicyID, SampleBatch]]): Optional
            dict of AgentIDs mapping to other agents' trajectory data (from the
            same episode). NOTE: The other agents use the same policy.
        episode (Optional[MultiAgentEpisode]): Optional multi-agent episode
            object in which the agents operated.

    Returns:
        SampleBatch: The postprocessed, modified SampleBatch (or a new one).
    """

    # postprocess of rewards
    status_dim = policy.model.status_dim
    emergency_dim = policy.model.emergency_feature_dim
    if other_agent_batches is not None:
        num_agents = len(other_agent_batches) + 1
    else:
        num_agents = 1

    # obs_shape = sample_batch[SampleBatch.OBS].shape
    agents_position = sample_batch[SampleBatch.OBS][..., num_agents + 2: num_agents + 4]
    # other_agents_relative_position = sample_batch[SampleBatch.OBS] \
    #     [..., num_agents + 4: num_agents + 4 + 2 * (num_agents - 1)].reshape(obs_shape[0], -1, 2)
    # if isinstance(other_agents_relative_position, np.ndarray):
    #     other_agents_relative_position = torch.from_numpy(other_agents_relative_position)
    # distance_between_agents = torch.norm(other_agents_relative_position, dim=2)
    try:
        emergency_position = sample_batch[VIRTUAL_OBS][..., status_dim:status_dim + emergency_dim]
    except KeyError:
        emergency_position = sample_batch[SampleBatch.OBS][..., status_dim:status_dim + emergency_dim]
    emergency_states = sample_batch[SampleBatch.OBS][..., 154:190].reshape(-1, policy.model.emergency_count,
                                                                           emergency_feature_in_state)
    modify_batch_with_intrinsic(agents_position, emergency_position, emergency_states, sample_batch)
    return compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)


def modify_batch_with_intrinsic(agents_position, emergency_position, emergency_states, sample_batch,
                                emergency_threshold=0):
    intrinsic = calculate_intrinsic(agents_position, emergency_position, emergency_states, emergency_threshold)
    # intrinsic[torch.mean(distance_between_agents < 0.1) > 0] *= 1.5
    sample_batch[ORIGINAL_REWARDS] = deepcopy(sample_batch[SampleBatch.REWARDS])
    sample_batch[INTRINSIC_REWARDS] = intrinsic
    if isinstance(sample_batch[SampleBatch.REWARDS], torch.Tensor):
        sample_batch[SampleBatch.REWARDS] += torch.from_numpy(intrinsic)
    else:
        sample_batch[SampleBatch.REWARDS] += intrinsic


def calculate_intrinsic(agents_position: np.ndarray,
                        emergency_position: np.ndarray,
                        emergency_states: np.ndarray,
                        emergency_threshold: int,
                        fake=False,
                        anti_goal_distances: np.ndarray = None,
                        indices=None,
                        mode: str = 'none',
                        alpha=0.1):
    """
    calculate the intrinsic reward for each agent, which is the product of distance and aoi.
    """
    # mask out penalty where emergency_positions are (0,0), which indicates the agent is not in the emergency.
    if len(emergency_position.shape) == 2:
        distances = np.linalg.norm(agents_position - emergency_position, axis=1)
        intrinsic = calculate_single_intrinsic(agents_position, alpha, anti_goal_distances, distances,
                                               emergency_position, emergency_states, emergency_threshold,
                                               fake, mode, indices=indices)
    elif len(emergency_position.shape) == 3:
        emergency_num = emergency_position.shape[1]
        intrinsic = np.zeros((len(agents_position), emergency_num))
        for i in range(emergency_num):
            if i == 0:
                last_pos = agents_position
            else:
                last_pos = emergency_position[:, i - 1]
            distances = np.linalg.norm(last_pos - emergency_position[:, i], axis=1)
            intrinsic[:, i] = c
            alculate_single_intrinsic(last_pos, alpha, None, distances,
                                                         emergency_position[:, i],
                                                         emergency_states, emergency_threshold, fake, mode)
        intrinsic = np.sum(intrinsic, axis=1)
        if anti_goal_distances is not None:
            intrinsic = intrinsic * (1 - alpha) + anti_goal_distances * alpha
    return intrinsic


def calculate_single_intrinsic(agents_position, alpha, anti_goal_distances, distances, emergency_position,
                               emergency_states, emergency_threshold, fake, mode, indices=None):
    mask = emergency_position.sum(axis=-1) != 0
    # find [aoi, (x,y)] array in state
    if not fake:
        if mode == 'none':
            intrinsic = np.zeros(len(agents_position))
        else:
            if mode == 'dis':
                intrinsic = -distances * mask
            else:
                age_of_informations, all_emergencies_position = emergency_states[..., 2], np.stack(
                    [emergency_states[..., 0], emergency_states[..., 1]], axis=-1)
                if indices is None:
                    indices = match_aoi(all_emergencies_position, emergency_position, mask)
                fetched_aois = age_of_informations[np.arange(len(age_of_informations)), indices]
                if mode == 'aoi':
                    intrinsic = -fetched_aois * mask
                elif mode == 'dis_aoi':
                    intrinsic = -distances * fetched_aois * mask
                elif mode == 'scaled_dis_aoi':
                    intrinsic = -distances * mask * fetched_aois / emergency_threshold
                    if anti_goal_distances is not None:
                        intrinsic = intrinsic * (1 - alpha) + anti_goal_distances * alpha
                else:
                    intrinsic = np.zeros(len(agents_position))
        # mask &= fetched_aois > emergency_threshold
    else:
        age_of_informations = np.arange(len(agents_position))
        # find the index of emergency with all_emergencies_position
        intrinsic = -distances * age_of_informations
    return intrinsic


@njit
def match_aoi(all_emergencies_position: np.ndarray,
              emergency_position: np.ndarray,
              mask: np.ndarray):
    indices = np.full((len(emergency_position)), dtype=np.int32, fill_value=-1)
    for i, this_timestep_pos in enumerate(emergency_position):
        if mask[i]:
            match_status = all_emergencies_position[i] == this_timestep_pos
            find_index = np.where(match_status[:, 0] & match_status[:, 1])[0]
            if len(find_index) > 0:
                indices[i] = find_index[0]
    return indices


def get_emergency_info(batch: SampleBatch, status_dim: int, emergency_dim: int):
    emergency_obs = batch[SampleBatch.OBS][..., status_dim:status_dim + emergency_dim]
    if isinstance(emergency_obs, torch.Tensor):
        emergency_obs = emergency_obs.cpu().numpy()
    start_indices, end_indices = get_emergency_start_end_numba(emergency_obs)
    selected_emergencies = emergency_obs[start_indices]
    return start_indices, end_indices, selected_emergencies


def add_auxiliary_loss(
        policy: TorchPolicy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    logging.debug("add_auxiliary_loss is called")
    status_dim = policy.model.status_dim
    my_device = model.device
    device = my_device
    num_agents = policy.model.n_agents
    this_emergency_count = policy.model.emergency_count
    emergency_dim = policy.model.emergency_dim
    emergency_feature_dim = policy.model.emergency_feature_dim
    selector = model.selector
    if hasattr(model, "selector_type") and 'NN' in model.selector_type and model.with_task_allocation:
        observation = train_batch[SampleBatch.OBS]
        try:
            observation[..., :status_dim + emergency_dim] = train_batch[VIRTUAL_OBS]
        except KeyError:
            pass
        inputs = observation[..., :status_dim + emergency_dim]
        labels = train_batch['labels']

        # test, greedy label (worked)
        # labels = torch.norm(train_batch[SampleBatch.OBS][..., 20:22] - train_batch[SampleBatch.OBS][..., 6:8], dim=-1)
        # filter out labels with -1
        mask = torch.logical_and(labels != -1, labels != 1)
        logging.debug(f"label count: {torch.sum(mask)}")
        if torch.sum(mask) == 0:
            mean_loss = torch.tensor(0.0)
        else:
            mean_loss = torch.tensor(0.0).to(my_device)
            if torch.sum(inputs[mask]) != 0:
                batch_size = 32
                learning_rate = 0.001 if model.render is False else 0
                epochs = 1
                criterion = nn.MSELoss()
                optimizer = optim.Adam(selector.parameters(), lr=learning_rate)
                dataset = torch.utils.data.TensorDataset(inputs[mask].to(torch.float32), labels[mask].to(torch.float32))
                mean_loss = train_predictor(batch_size, criterion, dataset, epochs, my_device, optimizer, selector)
                logging.debug(f"mean loss: {mean_loss}")
    else:
        mean_loss = torch.tensor(0.0)
    model.tower_stats['mean_regress_loss'] = mean_loss

    if hasattr(model, "use_aim") and model.use_aim and model.with_task_allocation:
        discriminator: nn.Module = model.discriminator
        optimizer = model.optimizer
        discriminator.train()
        cur_obs = train_batch[SampleBatch.CUR_OBS][..., :status_dim + emergency_dim]
        next_obs = train_batch[SampleBatch.NEXT_OBS][..., :status_dim + emergency_dim]
        cur_valid_mask = cur_obs[..., status_dim:status_dim + emergency_dim].sum(axis=-1) != 0
        next_valid_mask = next_obs[..., status_dim:status_dim + emergency_dim].sum(axis=-1) != 0
        valid_mask = torch.logical_and(cur_valid_mask, next_valid_mask)
        if torch.any(valid_mask):
            logging.debug("valid mask count: %d", torch.sum(valid_mask))
            optimizer.zero_grad()
            valid_cur_obs = cur_obs[valid_mask]
            cur_obs_potential = discriminator(valid_cur_obs)
            next_obs_potential = discriminator(next_obs[valid_mask])
            target_obs = valid_cur_obs.clone().detach()
            # add gaussian noise with mean=0, variance of 0.02
            target_pos = valid_cur_obs[:, status_dim:status_dim + emergency_dim]
            noise = torch.normal(mean=0, std=0.02, size=target_pos.shape).to(device)
            target_obs[:, num_agents + 2: num_agents + 4] = target_pos + noise
            target_obs_potential = discriminator(target_obs)
            new_preds = torch.cat([target_obs_potential, next_obs_potential], dim=0)
            model.reward_max = torch.max(new_preds).detach().cpu().numpy() + 0.1
            model.reward_min = torch.min(new_preds).detach().cpu().numpy() - 0.1
            penalty = 10 * torch.max(torch.abs(next_obs_potential - cur_obs_potential) - 0.1,
                                     torch.tensor(0.0).to(device))
            wgan_loss = (cur_obs_potential - target_obs_potential + penalty).mean()
            wgan_loss.backward()
            optimizer.step()
        else:
            wgan_loss = torch.tensor(0.0)
        model.tower_stats['mean_aim_loss'] = wgan_loss
    if hasattr(model, "NN_buffer") and hasattr(model, "prioritized_buffer") and \
            model.NN_buffer and model.prioritized_buffer and len(model.generator_inputs) > 0:
        generator = model.weight_generator
        batch_size = 32
        epochs = 1
        criterion = nn.MSELoss()
        inputs = torch.from_numpy(np.vstack(model.generator_inputs))
        labels = torch.from_numpy(np.concatenate(model.generator_labels))
        dataset = torch.utils.data.TensorDataset(inputs.to(torch.float32), labels.to(torch.float32))
        logging.debug(f"generator inputs shape: {inputs.shape}")
        mean_loss = train_predictor(batch_size, criterion, dataset, epochs,
                                    my_device, model.weight_optimizer, generator)
        logging.debug("mean loss of weight generator: %f", mean_loss)
        model.generator_inputs.clear()
        model.generator_labels.clear()
    model.tower_stats['mean_weight_loss'] = mean_loss
    lower_agent_batches = train_batch.split_by_episode()
    logging.debug("Switch Step Status: %s", model.step_count > model.switch_step)
    if hasattr(model, "selector_type") and 'RL' in model.selector_type and model.with_task_allocation:
        mean_reward = torch.tensor(0.0).to(device)
        if model.step_count > model.switch_step:
            rl_optimizer = model.high_level_optim
            assert 0 <= model.rl_gamma <= 1, "gamma should be in [0, 1]"
            selector.train()
            full_batches = []
            if model.last_rl_transitions is not None:
                for transitions in model.last_rl_transitions[model.train_count * 10:(model.train_count + 1) * 10]:
                    if len(transitions) > 0:
                        full_batches.append(SampleBatch.concat_samples(transitions))
            length_of_batches = len(full_batches)
            if length_of_batches > 0:
                # update assignment rl trajectory with lower level agent trajectories
                # episode_length = policy.model.episode_length
                # print("Number of batches: ", len(lower_agent_batches))
                for assign_agent_batch, lower_agent_batch in zip(full_batches, lower_agent_batches):
                    emergency_states = get_emergency_state(emergency_dim, lower_agent_batch, num_agents, status_dim,
                                                           this_emergency_count)
                    calculate_assign_rewards_lite(model, assign_agent_batch, lower_agent_batch, emergency_states,
                                                  status_dim,
                                                  emergency_feature_dim, num_agents,
                                                  agents_feature=(policy.model.selector_input_dim - 2) // num_agents,
                                                  mode=policy.model.reward_mode,
                                                  emergency_threshold=policy.model.emergency_threshold,
                                                  fail_hint=policy.model.fail_hint)
                    if model.use_relabeling != 'none':
                        new_batch = relabel_assign_batch(assign_agent_batch, model.relabel_threshold,
                                                         num_agents, 2, mode=model.use_relabeling)
                        if len(new_batch) > 0:
                            full_batches.append(new_batch)

                # progress = tqdm(range(1000))
                # for _ in progress:
                pcgrad = model.use_pcgrad
                mean_loss = torch.tensor(0.0).to(device)
                for batch in full_batches:
                    discounted_rewards = discount_cumsum(batch[SampleBatch.REWARDS], model.rl_gamma)
                    # normalize rewards
                    # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    #         discounted_rewards.std() + 1e-8)
                    # do not delete the copy(), discount_cumsum reverse the step for array.
                    discounted_rewards = torch.from_numpy(discounted_rewards.copy()).to(device)
                    mean_reward += torch.mean(discounted_rewards)
                    if model.use_neural_ucb:
                        selector.update_batch(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS],
                                              batch[SampleBatch.REWARDS])
                        selector.train()
                        mean_loss += model.selector.last_loss
                    else:
                        inputs = torch.from_numpy(batch[SampleBatch.CUR_OBS]).to(device)
                        invalid_masks = torch.from_numpy(batch['invalid_mask']).to(device)
                        if model.rl_use_cnn:
                            grids = torch.from_numpy(batch[Constants.IMAGE_STATE]).to(device)
                            batch_dist: Categorical = Categorical(probs=selector(inputs, invalid_mask=invalid_masks,
                                                                                 grid=grids))
                        else:
                            batch_dist: Categorical = Categorical(probs=selector(inputs, invalid_mask=invalid_masks))
                        actions_tensor = torch.from_numpy(batch[SampleBatch.ACTIONS]).to(device)
                        log_probs = batch_dist.log_prob(actions_tensor)
                        if pcgrad:
                            losses = (-log_probs * discounted_rewards).to(device)
                            rl_optimizer.pc_backward(losses)
                            mean_loss += losses.mean().detach()
                            rl_optimizer.step()
                        else:
                            loss = torch.mean(-log_probs * discounted_rewards).to(device)
                            rl_optimizer.zero_grad()
                            loss.backward()
                            mean_loss += loss.detach()
                            rl_optimizer.step()
                    # print gradient of the model
                mean_reward /= length_of_batches
                mean_loss /= length_of_batches
                # progress.set_postfix({'loss': mean_loss.item(), 'mean_reward': mean_reward.item()})
                selector.eval()
        if model.train_count == model.rl_update_interval - 1:
            model.train_count = 0
            model.last_rl_transitions = [[] for _ in range(model.num_envs)]
        else:
            if not torch.all(train_batch[SampleBatch.REWARDS] == 0):
                model.train_count += 1
        logging.debug(f"train function is called {model.train_count} times")
        logging.debug(f"RL Mean Loss: {mean_loss.item()}")
        logging.debug(f"RL Mean Reward: {mean_reward.item()}")
        model.tower_stats['mean_rl_loss'] = mean_loss
        model.tower_stats['mean_' + RAW_ASSIGN_REWARDS] = mean_reward

    if model.with_task_allocation:
        model.tower_stats['mean_' + INTRINSIC_REWARDS] = torch.mean(train_batch[INTRINSIC_REWARDS])
    model.train()
    total_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    if model.use_gdan and (not model.use_gdan_no_loss):
        logging.debug("Training Goal Discriminator")
        # TODO: missing *2 at here
        if len(model.goal_storage) > model.goal_batch_size * 2:
            aux_loss, accuracy = model.train_goal_discriminator()
            total_loss += model.gdan_eta * aux_loss
        else:
            aux_loss = torch.tensor(0.0)
            accuracy = torch.tensor(0.0)
        logging.debug(f"GDAN Loss: {aux_loss.item()}, Accuracy: {accuracy}")
        model.tower_stats['mean_gdan_loss'] = aux_loss
        model.tower_stats['gdan_accuracy'] = accuracy
        model.tower_stats['buffer_size'] = torch.tensor(len(model.goal_storage), dtype=torch.float32)
    # print the gradient of the model.p_encoder (query, keys, values)
    # for name, param in model.p_encoder.named_parameters():
    #     if param.grad is not None:
    #         logging.debug(f"Name: {name}, Gradient: {param.grad.mean()}")
    #     else:
    #         logging.debug(f"Name: {name}, Gradient: None")
    model.eval()
    return total_loss


def train_predictor(batch_size, criterion, dataset, epochs, my_device, optimizer, selector):
    mean_loss = torch.tensor(0.0).to(my_device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    progress = tqdm(range(epochs))
    selector.train()
    length = len(dataloader)
    for _ in progress:
        batch_loss = torch.tensor(0.0).to(my_device)
        for batch_observations, batch_distances in dataloader:
            optimizer.zero_grad()
            outputs = selector(batch_observations.to(my_device))
            if len(outputs.shape) > 1:
                outputs = outputs.squeeze(-1)
            loss = criterion(outputs, batch_distances.to(my_device))
            loss.backward()
            batch_loss += loss.detach()
            optimizer.step()
        progress.set_postfix({'mean_loss': batch_loss.item() / length})
        mean_loss += batch_loss
    mean_loss /= epochs * length
    selector.eval()
    return mean_loss



def get_emergency_state(emergency_dim, lower_agent_batch, num_agents, status_dim, this_emergency_count):
    observation = lower_agent_batch[SampleBatch.OBS]
    observation_dim = status_dim + emergency_dim + 100
    state_agents_dim = (num_agents + 4) * num_agents
    emergency_states = (observation[..., observation_dim + state_agents_dim:
                                         state_agents_dim + observation_dim +
                                         this_emergency_count * emergency_feature_in_state])
    emergency_states = emergency_states.reshape(-1, this_emergency_count, emergency_feature_in_state)
    return emergency_states


# @njit
def calculate_assign_rewards(allocated_emergencies, allocation_list, assign_rewards, end_indices, episode_length,
                             execute_rewards, start_indices, emergency_states, actions):
    assignment_status = emergency_states[..., 4][-1]
    emergency_xy = emergency_states[..., 0:2][-1]
    for i, (emergency, emergency_start, emergency_end) in enumerate(
            zip(allocated_emergencies, start_indices, end_indices)):
        allocate_index = np.nonzero((emergency[0] == allocation_list[:, 0]) &
                                    (emergency[1] == allocation_list[:, 1]) &
                                    (assign_rewards != -2))[0]
        delta = emergency_end - emergency_start
        # logging.debug(f"emergency_start: {emergency_start}, emergency_end: {emergency_end}")
        # if emergency_end % episode_length < episode_length - 1:
        # logging.debug(f"detected end: {emergency_end % episode_length}")
        trajectory_reward = execute_rewards[emergency_start:emergency_end]
        surveillance_reward = np.sum(trajectory_reward - np.floor(trajectory_reward))
        # Emergency cover reward consists of two parts, one is the emergency cover reward
        # and the other is the distance discount factor. The closer to the emergency
        # the higher the reward
        discount_factor = (episode_length - delta) / episode_length
        emergency_index = np.nonzero((emergency[0] == emergency_xy[:, 0]) &
                                     (emergency[1] == emergency_xy[:, 1]))[0]
        assert len(emergency_index) == 1
        if assignment_status[emergency_index] == actions[i]:
            emergency_cover_reward = EMERGENCY_REWARD_INCREMENT
        else:
            emergency_cover_reward = -1
        assign_rewards[allocate_index] = discount_factor * (surveillance_reward + emergency_cover_reward)
    # logging.debug(f"rewards for assignment agent: {assign_rewards}")


def generate_new_points(x: np.ndarray, y: np.ndarray, distances: Union[List, np.ndarray]):
    """
    Generate new points around the given (x, y) coordinates with the given distances.
    x: x array for all points
    y: y array for all points
    distances: the distances for each (x,y) pair.
    """
    # Number of points to generate, inferred from the length of distances array
    n = len(distances)
    # Convert distances list to a numpy array if not already
    distances = np.array(distances)
    # Generate n random angles
    angles = np.random.uniform(0, 2 * np.pi, n)
    # Calculate the new points' coordinates
    x_primes = x + distances * np.cos(angles)
    y_primes = y + distances * np.sin(angles)
    return np.vstack((x_primes, y_primes)).T


def relabel_assign_batch(assign_agent_batch: SampleBatch, relabel_threshold: float,
                         num_agents: int, emergency_feature_number: int,
                         mode: str = 'agent') -> SampleBatch:
    """
    relabel an episode of assign agent.
    assign_agent_batch: the SampleBatch for a assignment agent in an episode.
    relabel_threshold: any transition that achieved reward below relabel_threshold will receive a hindsight
    relabeled version.
    num_agents: number of lower level agents.
    emergency_feature_number: number of features for emergency in observation of assignment agents.
    mode: when mode is 'agent', the position of lower level agent selected by high level agent is modified to
    make assignment agent's action optimal. When mode is 'emergency', the position of emergency of that transition
    is modified so that assignment action is optimal.
    """
    assert RAW_ASSIGN_REWARDS in assign_agent_batch and SURVEILLANCE_REWARDS in assign_agent_batch
    relabel_mask = assign_agent_batch[RAW_ASSIGN_REWARDS] <= relabel_threshold
    if np.any(relabel_mask):
        new_batch = assign_agent_batch.copy()
        for key in new_batch.keys():
            new_batch[key] = new_batch[key][relabel_mask]
        new_batch[SampleBatch.REWARDS] = new_batch[SURVEILLANCE_REWARDS]
        new_length = new_batch.count = len(new_batch[SampleBatch.OBS])
        observations = new_batch[SampleBatch.OBS]
        actions = new_batch[SampleBatch.ACTIONS]
        agents_features, emergency_pos = create_optimal_label(actions, emergency_feature_number, mode, new_length,
                                                              num_agents, observations)
        # reassemble the observation (Is this the same pointer?)
        new_batch[SampleBatch.OBS] = np.concatenate([agents_features, emergency_pos], axis=-1, dtype=np.float32)
    else:
        new_batch = SampleBatch()
    return new_batch


def create_optimal_label(actions, emergency_feature_number, mode, new_length, num_agents, observations):
    emergency_pos, agents_features = observations[..., -emergency_feature_number:], observations[...,
                                                                                    :-emergency_feature_number]
    # new_agent_features = agents_features.copy()
    agents_pos = agents_features.reshape(new_length, num_agents, -1)[..., :2]
    distances = np.linalg.norm(agents_pos - emergency_pos[:, None, :], axis=-1)
    # find minimum distance for each agent
    min_distance = np.min(distances, axis=-1)
    # select agent_pos using actions
    if mode == 'agent':
        # create a mock agent (x,y) coordinate that has a norm smaller than min_distance in that batch
        agents_pos[np.arange(new_length), actions] = generate_new_points(emergency_pos[..., 0],
                                                                         emergency_pos[..., 1],
                                                                         min_distance)
    elif mode == 'emergency':
        # create a mock emergency (x,y) coordinate that has a norm smaller than min_distance in that batch
        selected_agent_pos = agents_pos[np.arange(new_length), actions]
        emergency_pos = generate_new_points(selected_agent_pos[..., 0],
                                            selected_agent_pos[..., 1],
                                            min_distance)
    else:
        raise NotImplementedError(f"mode: {mode} is not supported.")
    return agents_features, emergency_pos


def calculate_assign_rewards_lite(model: ModelV2,
                                  assign_agent_batch: SampleBatch, lower_agent_batch: SampleBatch,
                                  emergency_states: np.ndarray, status_dim: int, emergency_feature_dim: int,
                                  n_agents: int, agents_feature: int, emergency_threshold: int,
                                  mode: str = 'mix', fail_hint: bool = False):
    assign_obs, assign_rewards = assign_agent_batch[SampleBatch.OBS], assign_agent_batch[SampleBatch.REWARDS]
    surveillance_rewards = np.zeros_like(assign_rewards)
    raw_assign_rewards = np.zeros_like(assign_rewards)
    assign_actions = assign_agent_batch[SampleBatch.ACTIONS]
    logging.debug("assign rewards lite is called")
    if mode == 'mix':
        lower_level_rewards = lower_agent_batch[SampleBatch.REWARDS]
    elif mode == 'intrinsic':
        lower_level_rewards = lower_agent_batch[INTRINSIC_REWARDS]
    elif mode == 'original':
        lower_level_rewards = lower_agent_batch[ORIGINAL_REWARDS]
    elif mode == 'none' or mode == 'greedy':
        lower_level_rewards = np.zeros_like(assign_actions)
    else:
        raise ValueError(f"mode: {mode} is not supported.")
    emergency_in_lower_level_obs = lower_agent_batch[SampleBatch.OBS][...,
                                   status_dim:status_dim + emergency_feature_dim]
    if isinstance(emergency_in_lower_level_obs, torch.Tensor):
        emergency_in_lower_level_obs = emergency_in_lower_level_obs.cpu().numpy()
    if isinstance(lower_level_rewards, torch.Tensor):
        lower_level_rewards = lower_level_rewards.cpu().numpy()
    allocation_list = assign_obs[..., -2:]
    agents_pos = assign_obs[..., :-2].reshape(-1, n_agents, agents_feature)[..., 0:2]
    fraction = 1 / n_agents
    # fail_penalty = -EMERGENCY_REWARD_INCREMENT // 10
    if mode == 'greedy':
        for i, (action, emergency) in enumerate(zip(assign_actions, allocation_list)):
            distances = np.linalg.norm(agents_pos[i] - emergency, axis=1)
            discount_factor = np.zeros(n_agents)
            discount_factor[np.argsort(distances)] = np.linspace(fraction, 1, n_agents)
            emergency_cover_reward = 1 + fraction - discount_factor[action]
            assign_rewards[i] = emergency_cover_reward
            surveillance_rewards[i] = 1
            raw_assign_rewards[i] = emergency_cover_reward
    else:
        timesteps = assign_agent_batch['timesteps']
        rewards_by_agent = lower_level_rewards.reshape(-1, n_agents)
        for i, (current_time, action, emergency) in enumerate(zip(timesteps, assign_actions, allocation_list)):
            # discount_factor = (episode_length - delta) / episode_length
            distances = np.linalg.norm(agents_pos[i] - emergency, axis=1)
            discount_factor = np.zeros(n_agents)
            discount_factor[np.argsort(distances)] = np.linspace(fraction, 1, n_agents)
            if mode == 'none':
                emergency_cover_reward = 0
            else:
                mean_reward = np.mean(rewards_by_agent[current_time:current_time + emergency_threshold, action])
                model.reward_min = min(model.reward_min, mean_reward)
                model.reward_max = max(model.reward_max, mean_reward)
                # scale the reward according to reward_min and reward_max
                if model.reward_max - model.reward_min > 0:
                    mean_reward = (mean_reward - model.reward_min) / (model.reward_max - model.reward_min)
                if mean_reward > 0:
                    emergency_cover_reward = mean_reward * (1 + fraction - discount_factor[action])
                else:
                    emergency_cover_reward = mean_reward * discount_factor[action]
                logging.debug(f"mean_reward: {mean_reward}, with_emergency: {emergency_cover_reward}")
            assign_rewards[i] = emergency_cover_reward
            surveillance_rewards[i] = mean_reward
            raw_assign_rewards[i] = 1 + fraction - discount_factor[action]
    assign_agent_batch[SURVEILLANCE_REWARDS] = surveillance_rewards
    assign_agent_batch[RAW_ASSIGN_REWARDS] = raw_assign_rewards


def add_regress_loss_old(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # regression training of predictor
    if hasattr(model, "selector_type") and 'NN' in model.selector_type:
        virtual_obs = train_batch[VIRTUAL_OBS]
        (mean_relabel_percentage, predictor_inputs, predictor_labels,
         relabel_targets, train_batch_device) = increasing_intrinsic_relabeling(
            model, train_batch, virtual_obs)
        len_labels = len(predictor_labels)
        mean_relabel_percentage = mean_relabel_percentage / len_labels if len_labels > 0 else torch.tensor(0.0)
        if len_labels > 0:
            predictor_inputs = torch.stack(predictor_inputs).to(train_batch_device)
            predictor_labels = torch.tensor(predictor_labels).float().to(train_batch_device)
            regress_result = model.selector(predictor_inputs).squeeze(-1)
            # L2 Loss
            regress_loss = torch.nn.MSELoss()(regress_result, predictor_labels).to(model.device)
        else:
            regress_loss = torch.tensor(0.0).to(model.device)
        mean_label_count = torch.tensor(len(predictor_inputs), dtype=torch.float32)
        mean_valid_fragment_length = mean_length(relabel_targets)
    else:
        regress_loss = mean_relabel_percentage = torch.tensor(0.0).to(model.device)
        mean_label_count = mean_valid_fragment_length = torch.tensor(0.0, dtype=torch.float32)

    mean_regress_loss = torch.mean(regress_loss)
    model.train()
    total_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    regress_weight = 1.0
    total_loss += regress_weight * mean_regress_loss.to(total_loss.device)
    model.eval()
    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["total_loss"] = total_loss
    for item in ['mean_regress_loss', 'mean_label_count',
                 'mean_valid_fragment_length', 'mean_relabel_percentage']:
        model.tower_stats[item] = locals()[item]
    return total_loss


def increasing_intrinsic_relabeling(model, train_batch, virtual_obs):
    relabel_targets = find_ascending_sequences(train_batch[INTRINSIC_REWARDS])
    num_agents = 4
    predictor_inputs = []
    predictor_labels = []
    mean_relabel_percentage = torch.tensor(0.0)
    train_batch_device = train_batch[INTRINSIC_REWARDS].device
    for fragment in relabel_targets:
        obs_to_be_relabeled = virtual_obs[fragment]
        agent_position = obs_to_be_relabeled[..., num_agents + 2: num_agents + 4].to(model.device)
        emergency_position = obs_to_be_relabeled[..., 20:22].to(model.device)
        # ((250/7910)^2+(250/6960)^2)^0.5 = 0.048, the coverage distance example
        if torch.norm(emergency_position[-1] - agent_position[-1]) > 0.05:
            mean_relabel_percentage += 1
            # relabel needed
            emergency_position = agent_position[-1].repeat(len(emergency_position), 1).to(model.device)
            obs_to_be_relabeled[..., 20:22] = agent_position[-1]
            train_batch[INTRINSIC_REWARDS][fragment] = calculate_intrinsic(
                agent_position, emergency_position, torch.zeros(1, device=model.device),
                fake=True, emergency_threshold=0).to(torch.float32).to(train_batch_device)
            train_batch[SampleBatch.REWARDS][fragment] = (train_batch[ORIGINAL_REWARDS][fragment] +
                                                          train_batch[INTRINSIC_REWARDS][fragment])
            # additional complete bonus
            train_batch[SampleBatch.REWARDS][fragment[-1]] += len(fragment) * 6 / episode_length
            train_batch[VIRTUAL_OBS][fragment] = obs_to_be_relabeled
        # for successful trajectories, the aoi is not 100% correct (smaller than actual), check if needed.
        predictor_inputs.append(obs_to_be_relabeled[0])
        predictor_labels.append(len(fragment) / episode_length)
    return mean_relabel_percentage, predictor_inputs, predictor_labels, relabel_targets, train_batch_device


def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    """Attach virtual obs to sample batch for Intrinsic reward calculation."""
    model.last_sample_batch = input_dict
    extra_dict = vf_preds_fetches(policy, input_dict, state_batches, model, action_dist)
    if hasattr(model, "with_task_allocation") and model.with_task_allocation:
        extra_dict[VIRTUAL_OBS] = model.last_virtual_obs.cpu().numpy()
        if model.local_mode:
            extra_dict['buffer_indices'] = model.last_buffer_indices
            extra_dict['buffer_priority'] = model.last_buffer_priority
        if hasattr(model, "sibling_rivalry") and model.sibling_rivalry:
            extra_dict[ANTI_GOAL_REWARD] = model.last_anti_goal_reward
        for item in ['weight_matrix', 'selection', 'executor_obs']:
            model_item_name = 'last_' + item
            if hasattr(model.p_encoder, model_item_name):
                result = getattr(model.p_encoder, model_item_name)
                if result is not None:
                    extra_dict[item] = result
        if model.use_gdan_lstm:
            extra_dict['hx'] = model.main_att_encoder.last_hx
            extra_dict['cx'] = model.main_att_encoder.last_cx
    return extra_dict


def kl_and_loss_stats_with_regress(policy: TorchPolicy,
                                   train_batch: SampleBatch) -> Dict[str, TensorType]:
    """
    Add regress loss stats to the original stats.
    """
    original_dict = kl_and_loss_stats(policy, train_batch)
    #  'label_count', 'valid_fragment_length', 'relabel_percentage'
    if policy.model.with_task_allocation:
        for item in ['regress_loss', INTRINSIC_REWARDS, 'weight_loss']:
            original_dict[item] = torch.mean(torch.stack(policy.get_tower_stats("mean_" + item)))
        for model in policy.model_gpu_towers:
            if 'RL' in model.selector_type:
                original_dict[RAW_ASSIGN_REWARDS] = torch.mean(
                    torch.stack(policy.get_tower_stats("mean_" + RAW_ASSIGN_REWARDS)))
                original_dict['assign_reward_max'] = model.reward_max
                original_dict['assign_reward_min'] = model.reward_min
                original_dict['rl_loss'] = torch.mean(torch.stack(policy.get_tower_stats("mean_rl_loss")))
            if model.last_emergency_mode is not None:
                for i in range(model.n_agents):
                    original_dict[f'agent_{i}_mode'] = model.last_emergency_mode[i]
                    original_dict[f'agent_{i}_target_x'] = model.last_emergency_target[i][0]
                    original_dict[f'agent_{i}_target_y'] = model.last_emergency_target[i][1]
                    original_dict[f'agent_{i}_queue_length'] = model.last_emergency_queue_length[i]
            if hasattr(model, "intrinsic_mode") and model.intrinsic_mode == 'aim':
                original_dict['aim_loss'] = torch.mean(torch.stack(policy.get_tower_stats("mean_aim_loss")))
            if hasattr(model, 'last_selection') and model.last_selection is not None:
                original_dict[f'final_selection'] = model.last_selection
            if hasattr(model, "last_weight_matrix") and model.last_weight_matrix is not None:
                length = len(model.last_weight_matrix)
                for i in range(length):
                    original_dict[f'buffer_weight_{i}'] = model.last_weight_matrix[i]
            if model.use_gdan and (not model.use_gdan_no_loss):
                original_dict['gdan_loss'] = torch.mean(torch.stack(policy.get_tower_stats("mean_gdan_loss")))
                original_dict['gdan_accuracy'] = torch.mean(torch.stack(policy.get_tower_stats("gdan_accuracy")))
                original_dict['gdan_buffer_size'] = torch.mean(torch.stack(policy.get_tower_stats("buffer_size")))
                # log gradient mean into original_dict
                # for name, param in model.main_att_encoder.named_parameters():
                #     if param.grad is not None:
                #         original_dict[f"gradient_{name}"] = param.grad.mean()
                #     else:
                #         original_dict[f"gradient_{name}"] = 0
                # break
    return original_dict


def ppo_surrogate_loss_debug(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    # print(train_batch[SampleBatch.REWARDS].shape)
    # print(train_batch[SampleBatch.REWARDS][119::120])
    return ppo_surrogate_loss(policy, model, dist_class, train_batch)


def save_sample_batch(
        policy: TorchPolicy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    datatime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    all_agent_batches = {'drones_' + str(sample_batch['agent_index'][0]): sample_batch}
    if other_agent_batches is not None:
        for agent_id, (this_policy, batch) in other_agent_batches.items():
            all_agent_batches[agent_id] = batch
    for agent_id, batch in all_agent_batches.items():
        filename = f"output_{datatime_str}_agent_{agent_id}.pkl"
        with open(os.path.join("/workspace", "saved_data", filename), "wb") as pickle_file:
            logging.debug(f"saving trajectory file {filename}.")
            pickle.dump(batch, pickle_file)
    return compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)


def sample_batch_with_demonstration(
        policy: TorchPolicy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    # keep only numbers in sample_batch['agent_index']
    agent_id = re.sub(r"[^0-9]", "", str(int(sample_batch['agent_index'][0])))
    with open(os.path.join("/workspace", "saved_data", f"output_20240124-195236_agent_drones_{agent_id}.pkl"),
              "rb") as pickle_file:
        demonstration: SampleBatch = pickle.load(pickle_file)
        if random.random() < 0.5:
            logging.debug("using demonstration for agent_id: " + agent_id)
            return compute_gae_for_sample_batch(policy, demonstration, other_agent_batches, episode)
        else:
            logging.debug("using normal batch for agent_id: " + agent_id)
            return compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)


def after_loss_init(policy: Policy, observation_space: gym.spaces.Space,
                    action_space: gym.spaces.Space, config: TrainerConfigDict) -> None:
    policy.view_requirements[ORIGINAL_REWARDS] = ViewRequirement(
        ORIGINAL_REWARDS, shift=0, used_for_training=True)
    policy.view_requirements[INTRINSIC_REWARDS] = ViewRequirement(
        INTRINSIC_REWARDS, shift=0, used_for_training=True)





def get_policy_class_traffic_ppo(config_):
    if config_["framework"] == "torch":
        return TrafficPPOTorchPolicy


@njit
def get_emergency_start_end_numba(emergency_obs: np.ndarray):
    start_indices, end_indices = [], []
    length = len(emergency_obs)
    i = 0
    while i < length:
        entry = emergency_obs[i]
        if entry.any():
            start_indices.append(i)
            last_entry = entry
            while np.all(last_entry == entry):
                i += 1
                if i < length:
                    entry = emergency_obs[i]
                else:
                    break
            end_indices.append(i - 1)
        else:
            i += 1
    return start_indices, end_indices


def get_emergency_start_end(emergency_obs: np.ndarray):
    separate_results = list(np.where(np.any(np.diff(emergency_obs, axis=0) != 0, axis=1))[0] + 1)
    start_indices, end_indices = [], []
    length = len(separate_results)
    j = 0
    while j < length:
        start_indices.append(separate_results[j])
        if j < length - 1:
            expected_end = separate_results[j + 1]
            end_indices.append(expected_end - 1)
            if not np.any(emergency_obs[expected_end]):
                j += 2
            else:
                j += 1
        else:
            end_indices.append(len(emergency_obs) - 1)
            j += 1
    if np.any(emergency_obs[0]):
        # deal with the bug for np.diff ignoring prefix.
        if np.any(np.all(emergency_obs == 0, axis=1)):
            start_indices[0] = 0
            end_indices[0] = separate_results[0] - 1
            separate_results[0] = 0
        else:
            start_indices.insert(0, 0)
            end_indices.insert(0, separate_results[0] - 1)
            separate_results.insert(0, 0)
    return end_indices, start_indices, separate_results


TrafficPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="TrafficPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=relabel_for_sample_batch,
    loss_fn=add_auxiliary_loss,
    extra_action_out_fn=extra_action_out_fn,
    stats_fn=kl_and_loss_stats_with_regress,
    _after_loss_init=after_loss_init,
)

TrafficPPOTrainer = WandbPPOTrainer.with_updates(
    name="TRAFFICPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_traffic_ppo,
)
