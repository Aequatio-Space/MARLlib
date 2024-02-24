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
import gym
import numpy as np
from marllib.marl.algos.wandb_trainers import WandbPPOTrainer
from numba import njit
from ray.rllib.agents.ppo import PPOTorchPolicy, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import (kl_and_loss_stats, vf_preds_fetches,
                                                   ppo_surrogate_loss, compute_gae_for_sample_batch)
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
    use_intrinsic = True
    selector_type = policy.model.selector_type
    use_distance_intrinsic = 'NN' not in selector_type or policy.model.step_count < policy.model.switch_step
    # logging.debug("use_distance_intrinsic: %s", use_distance_intrinsic)
    # logging.debug("step_count: %d, switch_step: %d", policy.model.step_count, policy.model.switch_step)
    status_dim = policy.model.status_dim
    emergency_dim = policy.model.emergency_dim
    emergency_feature_dim = policy.model.emergency_feature_dim
    # postprocess of rewards
    num_agents = policy.model.n_agents
    # postprocess extra_batches
    try:
        virtual_obs = sample_batch[VIRTUAL_OBS]
        sample_batch[SampleBatch.OBS][..., :status_dim + emergency_dim] = virtual_obs
    except KeyError:
        pass
    extra_batches = [sample_batch.copy() for _ in range(k)]
    future_multi_goal_relabeling(extra_batches, goal_number, num_agents, sample_batch, policy)
    this_emergency_count = policy.model.emergency_count
    observation = sample_batch[SampleBatch.OBS]
    emergency_states = get_emergency_state(emergency_dim, sample_batch, num_agents, status_dim, this_emergency_count)
    agents_position = observation[..., num_agents + 2: num_agents + 4]
    num_envs = policy.model.num_envs
    if use_intrinsic:
        if policy.model.buffer_in_obs:
            emergency_position = observation[..., status_dim:status_dim + emergency_feature_dim]
        else:
            emergency_position = observation[..., status_dim:status_dim + emergency_dim]
        if use_distance_intrinsic:
            if policy.model.sibling_rivalry:
                assert ANTI_GOAL_REWARD in sample_batch, ANTI_GOAL_REWARD + " is not in sample_batch."
                anti_goal_distances = sample_batch[ANTI_GOAL_REWARD]
            else:
                anti_goal_distances = None
            intrinsic = calculate_intrinsic(agents_position, emergency_position, emergency_states,
                                            emergency_threshold=policy.model.emergency_threshold,
                                            anti_goal_distances=anti_goal_distances,
                                            alpha=policy.model.alpha)
        else:
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
        # intrinsic[torch.mean(distance_between_agents < 0.1) > 0] *= 1.5
        sample_batch[ORIGINAL_REWARDS] = deepcopy(sample_batch[SampleBatch.REWARDS])
        sample_batch['intrinsic_rewards'] = intrinsic
        if isinstance(sample_batch[SampleBatch.REWARDS], torch.Tensor):
            sample_batch[SampleBatch.REWARDS] += torch.from_numpy(intrinsic)
        else:
            sample_batch[SampleBatch.REWARDS] += intrinsic
    # return label_done_masks_and_calculate_gae_for_sample_batch(policy,
    #                                                            sample_batch,
    #                                                            other_agent_batches,
    #                                                            episode
    #                                                            )
    postprocess_batches = [label_done_masks_and_calculate_gae_for_sample_batch(policy, sample_batch,
                                                                               other_agent_batches, episode)]
    for batch in extra_batches:
        # must copy a batch since the input dict will be equipped with torch interceptor.
        _ = policy.compute_actions_from_input_dict(batch.copy())
        batch.update(policy.extra_action_out(batch, [], policy.model, None))
        if use_intrinsic:
            if use_large_emergency:
                # Extract non-zero elements
                emergency_matrix = batch[SampleBatch.OBS][..., status_dim:status_dim + emergency_dim]
                emergency_position = np.zeros_like(agents_position)
                for i in range(len(emergency_position)):
                    indices = np.nonzero(emergency_matrix[i])[0]
                    if len(indices) != 0:
                        emergency_position[i] = emergency_matrix[i][indices]
            else:
                emergency_matrix = emergency_position = batch[SampleBatch.OBS][...,
                                                        status_dim:status_dim + emergency_dim]
            # Reshape the array into the desired format
            if emergency_matrix.shape[0] != 0:
                modify_batch_with_intrinsic(agents_position, emergency_position, emergency_states, batch)
        # new_batch = compute_gae_for_sample_batch(policy, batch, other_agent_batches, episode)
        new_batch = label_done_masks_and_calculate_gae_for_sample_batch(policy, batch, other_agent_batches, episode)
        postprocess_batches.append(new_batch)
    return SampleBatch.concat_samples(postprocess_batches)
    # try, send relabeled trajectory only.


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

    # check whether numbers between status_dim:status_dim + emergency_dim are all zeros, one row per value
    emergency_obs = sample_batch[SampleBatch.OBS][..., status_dim:status_dim + emergency_dim]
    no_emergency_mask = np.nansum(np.abs(emergency_obs), axis=1) == 0
    emergencies_masks = []
    separate_batches = []
    last_index = 0
    try:
        raw_rewards = sample_batch['original_rewards']
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
    sample_batch['original_rewards'] = deepcopy(sample_batch[SampleBatch.REWARDS])
    sample_batch['intrinsic_rewards'] = intrinsic
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
                        alpha=0.1):
    """
    calculate the intrinsic reward for each agent, which is the product of distance and aoi.
    """
    # mask out penalty where emergency_positions are (0,0), which indicates the agent is not in the emergency.

    mask = emergency_position.sum(axis=-1) != 0
    distances = np.linalg.norm(agents_position - emergency_position, axis=1)
    # add antil gaol at here.
    # find [aoi, (x,y)] array in state
    if not fake:
        age_of_informations, all_emergencies_position = emergency_states[..., 2], np.stack(
            [emergency_states[..., 0], emergency_states[..., 1]], axis=-1)
        indices = match_aoi(all_emergencies_position, emergency_position, mask)
        fetched_aois = age_of_informations[np.arange(len(age_of_informations)), indices]
        mask &= fetched_aois > emergency_threshold
        if anti_goal_distances is not None:
            intrinsic = (anti_goal_distances * alpha - distances) * mask * fetched_aois
        else:
            intrinsic = -distances * mask * fetched_aois
    else:
        age_of_informations = np.arange(len(agents_position))
        # find the index of emergency with all_emergencies_position
        intrinsic = -distances * age_of_informations
    return intrinsic


@njit
def match_aoi(all_emergencies_position: np.ndarray,
              emergency_position: np.ndarray,
              mask: np.ndarray):
    indices = np.zeros((len(emergency_position)), dtype=np.int32)
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
    device = model.device
    num_agents = policy.model.n_agents
    this_emergency_count = policy.model.emergency_count
    emergency_dim = policy.model.emergency_dim
    emergency_feature_dim = policy.model.emergency_feature_dim
    if hasattr(model, "selector_type") and 'NN' in model.selector_type:
        batch_size = 32
        learning_rate = 0.001 if model.render is False else 0
        epochs = 1
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.selector.parameters(), lr=learning_rate)
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
            dataset = torch.utils.data.TensorDataset(inputs[mask].to(torch.float32), labels[mask].to(torch.float32))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            mean_loss = torch.tensor(0.0).to(model.device)
            if torch.sum(inputs) != 0:
                progress = tqdm(range(epochs))
                model.selector.train()
                for _ in progress:
                    for batch_observations, batch_distances in dataloader:
                        optimizer.zero_grad()
                        outputs = model.selector(batch_observations.to(model.device))
                        if len(outputs.shape) > 1:
                            outputs = outputs.squeeze()
                        loss = criterion(outputs, batch_distances.to(model.device))
                        loss.backward()
                        mean_loss += loss.detach()
                        optimizer.step()
                    progress.set_postfix({'mean_loss': mean_loss.item()})
                mean_loss /= epochs * len(dataloader)
                model.selector.eval()
    else:
        mean_loss = torch.tensor(0.0)
    model.tower_stats['mean_regress_loss'] = mean_loss
    lower_agent_batches = train_batch.split_by_episode()
    logging.debug("Switch Step Status: %s", model.step_count > model.switch_step)
    if hasattr(model, "selector_type") and 'RL' in model.selector_type:
        parameters_list = [item for item in model.selector.parameters()]
        mean_reward = torch.tensor(0.0).to(device)
        if len(parameters_list) > 0 and model.step_count > model.switch_step:
            rl_optimizer = optim.Adam(parameters_list, lr=0.001)
            assert 0 <= model.rl_gamma <= 1, "gamma should be in [0, 1]"
            model.selector.train()
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
                    calculate_assign_rewards_lite(assign_agent_batch, lower_agent_batch, emergency_states, status_dim,
                                                  emergency_feature_dim, num_agents, mode=policy.model.reward_mode,
                                                  fail_hint=policy.model.fail_hint)
                mean_loss = torch.tensor(0.0).to(device)
                # progress = tqdm(full_batches)
                for batch in full_batches:
                    discounted_rewards = discount_cumsum(batch[SampleBatch.REWARDS], model.rl_gamma)
                    # normalize rewards
                    # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    #         discounted_rewards.std() + 1e-8)
                    # do not delete the copy(), discount_cumsum reverse the step for array.
                    discounted_rewards = torch.from_numpy(discounted_rewards.copy()).to(device)
                    mean_reward += torch.mean(discounted_rewards)
                    inputs = torch.from_numpy(batch[SampleBatch.CUR_OBS]).to(device)
                    invalid_masks = torch.from_numpy(batch['invalid_mask']).to(device)
                    batch_dist: Categorical = Categorical(probs=model.selector(inputs, invalid_mask=invalid_masks))
                    actions_tensor = torch.from_numpy(batch[SampleBatch.ACTIONS]).to(device)
                    log_probs = batch_dist.log_prob(actions_tensor)
                    loss = torch.mean(-log_probs * discounted_rewards).to(device)
                    # progress.set_postfix({'loss': loss.item()})
                    rl_optimizer.zero_grad()
                    loss.backward()
                    mean_loss += loss.detach()
                    rl_optimizer.step()
                    # print gradient of the model
                mean_reward /= length_of_batches
                mean_loss /= length_of_batches
                model.selector.eval()
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
        model.tower_stats['mean_assign_reward'] = mean_reward
    model.tower_stats['mean_intrinsic_rewards'] = torch.mean(train_batch['intrinsic_rewards'])
    model.train()
    total_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    model.eval()
    return total_loss


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


def calculate_assign_rewards_lite(assign_agent_batch: SampleBatch, lower_agent_batch: SampleBatch,
                                  emergency_states: np.ndarray, status_dim: int, emergency_feature_dim: int,
                                  n_agents: int, mode: str = 'mix', fail_hint: bool = False):
    assign_obs, assign_rewards = assign_agent_batch[SampleBatch.OBS], assign_agent_batch[SampleBatch.REWARDS]
    assign_actions = assign_agent_batch[SampleBatch.ACTIONS]
    logging.debug("assign rewards lite is called")
    if mode == 'mix':
        lower_level_rewards = lower_agent_batch[SampleBatch.REWARDS]
    elif mode == 'intrinsic':
        lower_level_rewards = lower_agent_batch['intrinsic_rewards']
    elif mode == 'original':
        lower_level_rewards = lower_agent_batch['original_rewards']
    elif mode == 'none':
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
    agents_pos = assign_obs[..., :-2].reshape(-1, n_agents, 3)[..., 0:2]
    assignment_status = list(emergency_states[..., 4][-1])
    emergency_xy = emergency_states[..., 0:2][-1]
    start_indices, end_indices = get_emergency_start_end_numba(emergency_in_lower_level_obs)
    selected_emergencies = emergency_in_lower_level_obs[np.array(start_indices)]
    # construct a dict from emergency_xy
    emergency_dict = {(x, y): i for i, (x, y) in enumerate(emergency_xy)}
    lower_level_dict = {(x, y): i for i, (x, y) in enumerate(selected_emergencies)}
    fail_penalty = -1.0 if (mode == 'original' or mode == 'none') else -50.0
    for i, (action, emergency) in enumerate(zip(assign_actions, allocation_list)):
        coordinates_tuple = (emergency[0], emergency[1])
        emergency_index = emergency_dict.get(coordinates_tuple, -1)
        lower_level_index = lower_level_dict.get(coordinates_tuple, -1)
        if lower_level_index == -1:
            emergency_cover_reward = fail_penalty
        else:
            if assignment_status[emergency_index] == action:
                delta = end_indices[lower_level_index] - start_indices[lower_level_index]
                discount_factor = (episode_length - delta) / episode_length
                if mode == 'none':
                    mean_reward = 0.0
                else:
                    if delta > 0:
                        mean_reward = np.mean(lower_level_rewards[start_indices[lower_level_index]:
                                                                  end_indices[lower_level_index]])
                    else:
                        mean_reward = 0.0
                    if np.isnan(mean_reward):
                        mean_reward = 0.0
                emergency_cover_reward = (EMERGENCY_REWARD_INCREMENT + mean_reward) * discount_factor
            else:
                if fail_hint:
                    distance_factor = np.linalg.norm(agents_pos[i][action] - emergency)
                    emergency_cover_reward = fail_penalty * distance_factor
                else:
                    emergency_cover_reward = fail_penalty
        assign_rewards[i] = emergency_cover_reward


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
    relabel_targets = find_ascending_sequences(train_batch['intrinsic_rewards'])
    num_agents = 4
    predictor_inputs = []
    predictor_labels = []
    mean_relabel_percentage = torch.tensor(0.0)
    train_batch_device = train_batch['intrinsic_rewards'].device
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
            train_batch['intrinsic_rewards'][fragment] = calculate_intrinsic(
                agent_position, emergency_position, torch.zeros(1, device=model.device),
                fake=True, emergency_threshold=0).to(torch.float32).to(train_batch_device)
            train_batch[SampleBatch.REWARDS][fragment] = (train_batch['original_rewards'][fragment] +
                                                          train_batch['intrinsic_rewards'][fragment])
            # additional complete bonus
            train_batch[SampleBatch.REWARDS][fragment[-1]] += len(fragment) * 6 / episode_length
            train_batch[VIRTUAL_OBS][fragment] = obs_to_be_relabeled
        # for successful trajectories, the aoi is not 100% correct (smaller than actual), check if needed.
        predictor_inputs.append(obs_to_be_relabeled[0])
        predictor_labels.append(len(fragment) / episode_length)
    return mean_relabel_percentage, predictor_inputs, predictor_labels, relabel_targets, train_batch_device


def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    """Attach virtual obs to sample batch for Intrinsic reward calculation."""
    extra_dict = vf_preds_fetches(policy, input_dict, state_batches, model, action_dist)
    if hasattr(model, "with_task_allocation") and model.with_task_allocation:
        extra_dict[VIRTUAL_OBS] = model.last_virtual_obs.cpu().numpy()
    if hasattr(model, "sibling_rivalry") and model.sibling_rivalry:
        extra_dict[ANTI_GOAL_REWARD] = model.last_anti_goal_reward
    return extra_dict


def kl_and_loss_stats_with_regress(policy: TorchPolicy,
                                   train_batch: SampleBatch) -> Dict[str, TensorType]:
    """
    Add regress loss stats to the original stats.
    """
    original_dict = kl_and_loss_stats(policy, train_batch)
    #  'label_count', 'valid_fragment_length', 'relabel_percentage'
    for item in ['regress_loss', 'intrinsic_rewards']:
        original_dict[item] = torch.mean(torch.stack(policy.get_tower_stats("mean_" + item)))
    for model in policy.model_gpu_towers:
        if 'RL' in model.selector_type:
            original_dict['assign_reward'] = torch.mean(torch.stack(policy.get_tower_stats("mean_assign_reward")))
            original_dict['rl_loss'] = torch.mean(torch.stack(policy.get_tower_stats("mean_rl_loss")))
        if model.last_emergency_mode is not None:
            for i in range(model.n_agents):
                original_dict[f'agent_{i}_mode'] = model.last_emergency_mode[i]
                original_dict[f'agent_{i}_target_x'] = model.last_emergency_target[i][0]
                original_dict[f'agent_{i}_target_y'] = model.last_emergency_target[i][1]
                original_dict[f'agent_{i}_queue_length'] = model.last_emergency_queue_length[i]
        break
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
    policy.view_requirements['intrinsic_rewards'] = ViewRequirement(
        'intrinsic_rewards', shift=0, used_for_training=True)


TrafficPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="TrafficPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=relabel_for_sample_batch,
    loss_fn=add_auxiliary_loss,
    extra_action_out_fn=extra_action_out_fn,
    stats_fn=kl_and_loss_stats_with_regress,
    _after_loss_init=after_loss_init,
)


def get_policy_class_traffic_ppo(config_):
    if config_["framework"] == "torch":
        return TrafficPPOTorchPolicy


TrafficPPOTrainer = WandbPPOTrainer.with_updates(
    name="TRAFFICPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_traffic_ppo,
)


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
