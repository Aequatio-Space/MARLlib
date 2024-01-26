"""
PyTorch policy class used for PPO.
"""
import datetime
import os
import pickle
import random
import re

import gym
import logging
from typing import Dict, List, Type, Union, Optional
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
from marllib.marl.algos.wandb_trainers import WandbPPOTrainer
from ray.rllib.agents.ppo import PPOTorchPolicy, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import (kl_and_loss_stats, vf_preds_fetches,
                                                   ppo_surrogate_loss, compute_gae_for_sample_batch)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict, AgentID
from torch import optim

EMERGENCY_REWARD_INCREMENT = 10

status_dim = 20

emergency_features = 2

emergency_count = 9

emergency_dim = emergency_features

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

VIRTUAL_OBS = 'virtual_obs'

use_large_emergency = False

# dirty hack, should be removed later
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
    goal_number = 3
    use_intrinsic = True

    # postprocess of rewards
    num_agents = policy.model.n_agents
    # postprocess extra_batches
    try:
        virtual_obs = sample_batch[VIRTUAL_OBS]
        sample_batch[SampleBatch.OBS][..., :status_dim + emergency_dim] = virtual_obs
    except KeyError:
        pass
    observation_dim = policy.model.status_dim + policy.model.emergency_feature_dim + 100
    state_agents_dim = (num_agents + 4) * num_agents
    this_emergency_count = policy.model.emergency_count
    observation = sample_batch[SampleBatch.OBS]
    agents_position = observation[..., num_agents + 2: num_agents + 4]
    emergency_states = (observation[..., observation_dim + state_agents_dim:
                                         state_agents_dim + observation_dim + this_emergency_count * 4])
    emergency_states = emergency_states.reshape(-1, this_emergency_count, 4)
    extra_batches = [sample_batch.copy() for _ in range(k)]
    future_multi_goal_relabeling(extra_batches, goal_number, num_agents, sample_batch)
    # fix_interval_relabeling(extra_batches, 20, num_agents, sample_batch)
    emergency_position = sample_batch[SampleBatch.OBS][..., status_dim:status_dim + emergency_dim]
    if use_intrinsic:
        modify_batch_with_intrinsic(agents_position, emergency_position, emergency_states, sample_batch)
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
    # return postprocess_batches[-1]


def label_done_masks_and_calculate_gae_for_sample_batch(
        policy: TorchPolicy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    separate_batches = []
    last_index = 0
    # check whether numbers between status_dim:status_dim + emergency_dim are all zeros, one row per value
    no_emergency_mask = np.nansum(np.abs(sample_batch[SampleBatch.OBS][...,
                                         status_dim:status_dim + emergency_dim]), axis=1) == 0
    emergencies_masks = []
    try:
        raw_rewards = sample_batch['original_rewards']
    except KeyError:
        raw_rewards = sample_batch[SampleBatch.REWARDS]
    for index, reward in enumerate(raw_rewards):
        if reward >= EMERGENCY_REWARD_INCREMENT:
            # sample_batch[SampleBatch.DONES][index] = True
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
            batch['labels'] = labels
        else:
            batch['labels'] = np.full(batch_length,
                                      -1)  # discard last batch by default since it contains truncated result.
        labeled_batches.append(batch)
        # labeled_batches.append(compute_gae_for_sample_batch(policy, batch, other_agent_batches, episode))
    full_batch = SampleBatch.concat_samples(labeled_batches)
    return compute_gae_for_sample_batch(policy, full_batch, other_agent_batches, episode)


def fix_interval_relabeling(extra_batches, goal_number, num_agents, sample_batch):
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


def mapping_relabeling(extra_batches, start_timestep, num_agents, sample_batch):
    for i in range(len(extra_batches)):
        original_batch = extra_batches[i]
        obs_shape = original_batch[SampleBatch.OBS].shape
        agents_position = sample_batch[SampleBatch.OBS][..., num_agents + 2: num_agents + 2 + 2]
        goals_to_be_filled = agents_position[start_timestep:, :]
        # sample an ascending sequence of with length goal_number
        original_obs = original_batch[SampleBatch.OBS]
        original_obs[:len(original_obs) - start_timestep, status_dim:status_dim + emergency_dim] = 0
        original_obs[:len(original_obs) - start_timestep,
        status_dim: status_dim + emergency_features] = goals_to_be_filled
        extra_batches[i] = original_batch


def future_multi_goal_relabeling(extra_batches, goal_number, num_agents, sample_batch):
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
                original_obs[j][status_dim:status_dim + emergency_dim] = 0
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
        policy: Policy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory.

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
    if other_agent_batches is not None:
        num_agents = len(other_agent_batches) + 1
    else:
        num_agents = 1

    obs_shape = sample_batch[SampleBatch.OBS].shape
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
    emergency_states = sample_batch[SampleBatch.OBS][..., 154:190].reshape(-1, policy.model.emergency_count, 4)
    modify_batch_with_intrinsic(agents_position, emergency_position, emergency_states, sample_batch)
    return compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)


def modify_batch_with_intrinsic(agents_position, emergency_position, emergency_states, sample_batch):
    intrinsic = calculate_intrinsic(agents_position, emergency_position, emergency_states)
    # intrinsic[torch.mean(distance_between_agents < 0.1) > 0] *= 1.5
    sample_batch['original_rewards'] = deepcopy(sample_batch[SampleBatch.REWARDS])
    sample_batch['intrinsic_rewards'] = intrinsic
    if isinstance(sample_batch[SampleBatch.REWARDS], torch.Tensor):
        sample_batch[SampleBatch.REWARDS] += intrinsic
    else:
        sample_batch[SampleBatch.REWARDS] += intrinsic.cpu().numpy()


def calculate_intrinsic(agents_position: Union[torch.Tensor, np.ndarray],
                        emergency_position: Union[torch.Tensor, np.ndarray],
                        emergency_states: Union[torch.Tensor, np.ndarray], fake=False, device='cpu'):
    """
    calculate the intrinsic reward for each agent, which is the product of distance and aoi.
    """
    # mask out penalty where emergency_positions are (0,0), which indicates the agent is not in the emergency.
    if isinstance(agents_position, np.ndarray):
        agents_position = torch.from_numpy(agents_position)
    if isinstance(emergency_position, np.ndarray):
        emergency_position = torch.from_numpy(emergency_position)
    if isinstance(emergency_states, np.ndarray):
        emergency_states = torch.from_numpy(emergency_states)
    mask = emergency_position.sum(axis=-1) != 0
    distances = torch.norm(agents_position - emergency_position, dim=1).to(device)
    # find [aoi, (x,y)] array in state
    if not fake:
        age_of_informations, all_emergencies_position = emergency_states[..., 2], torch.stack(
            [emergency_states[..., 0], emergency_states[..., 1]], dim=-1)
    else:
        age_of_informations = torch.arange(len(agents_position), device=device).float()
    # find the index of emergency with all_emergencies_position
    if not fake:
        indices = torch.zeros((len(emergency_position)), dtype=torch.int32)
        for i, this_timestep_pos in enumerate(emergency_position):
            if mask[i]:
                find_index = torch.nonzero(torch.all(all_emergencies_position[i] == this_timestep_pos, dim=1)).squeeze(
                    -1)
                if len(find_index) > 0:
                    indices[i] = find_index[0]
        intrinsic = -distances * mask * age_of_informations[torch.arange(len(age_of_informations)), indices]
    else:
        intrinsic = -distances * age_of_informations
    return intrinsic


def add_regress_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    if hasattr(model, "selector_type") and model.selector_type == 'NN':
        batch_size = 32
        learning_rate = 0.001
        epochs = 1
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        try:
            train_batch[SampleBatch.OBS][..., :status_dim + emergency_dim] = train_batch[VIRTUAL_OBS]
        except KeyError:
            pass
        inputs = train_batch[SampleBatch.OBS][..., :status_dim + emergency_dim]
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
    model.train()
    total_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    model.eval()

    return total_loss


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
    if hasattr(model, "selector_type") and model.selector_type == 'NN':
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
                fake=True, device=model.device).to(torch.float32).to(train_batch_device)
            train_batch[SampleBatch.REWARDS][fragment] = (train_batch['original_rewards'][fragment] +
                                                          train_batch['intrinsic_rewards'][fragment])
            # additional complete bonus
            train_batch[SampleBatch.REWARDS][fragment[-1]] += len(fragment) * 5 / episode_length
            train_batch[VIRTUAL_OBS][fragment] = obs_to_be_relabeled
        # for successful trajectories, the aoi is not 100% correct (smaller than actual), check if needed.
        predictor_inputs.append(obs_to_be_relabeled[0])
        predictor_labels.append(len(fragment) / episode_length)
    return mean_relabel_percentage, predictor_inputs, predictor_labels, relabel_targets, train_batch_device


def extra_action_out_fn_old(policy, input_dict, state_batches, model, action_dist):
    """Attach virtual obs to sample batch for Intrinsic reward calculation."""
    extra_dict = vf_preds_fetches(policy, input_dict, state_batches, model, action_dist)
    try:
        extra_dict['virtual_obs'] = model.last_virtual_obs
        # extra_dict['predicted_values'] = model.last_predicted_values
    except AttributeError:
        pass
    return extra_dict


def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    """Attach virtual obs to sample batch for Intrinsic reward calculation."""
    extra_dict = vf_preds_fetches(policy, input_dict, state_batches, model, action_dist)
    try:
        extra_dict[VIRTUAL_OBS] = model.last_virtual_obs.cpu().numpy()
        # extra_dict['predicted_values'] = model.last_predicted_values
    except AttributeError:
        pass
    return extra_dict


def kl_and_loss_stats_with_regress(policy: Policy,
                                   train_batch: SampleBatch) -> Dict[str, TensorType]:
    """
    Add regress loss stats to the original stats.
    """
    original_dict = kl_and_loss_stats(policy, train_batch)
    #  'label_count', 'valid_fragment_length', 'relabel_percentage'
    for item in ['regress_loss']:
        original_dict[item] = torch.mean(torch.stack(policy.get_tower_stats("mean_" + item)))
    for model in policy.model_gpu_towers:

        if model.last_emergency_mode is not None:
            for i in range(model.n_agents):
                original_dict[f'agent_{i}_mode'] = model.last_emergency_mode[i]
                original_dict[f'agent_{i}_target_x'] = model.last_emergency_target[i][0]
                original_dict[f'agent_{i}_target_y'] = model.last_emergency_target[i][1]
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


TrafficPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="TrafficPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=relabel_for_sample_batch,
    loss_fn=add_regress_loss,
    extra_action_out_fn=extra_action_out_fn,
    stats_fn=kl_and_loss_stats_with_regress,
)


def get_policy_class_traffic_ppo(config_):
    if config_["framework"] == "torch":
        return TrafficPPOTorchPolicy


TrafficPPOTrainer = WandbPPOTrainer.with_updates(
    name="TRAFFICPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_traffic_ppo,
)
