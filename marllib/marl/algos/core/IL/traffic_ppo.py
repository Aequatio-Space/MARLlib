"""
PyTorch policy class used for PPO.
"""
import gym
import logging
from typing import Dict, List, Type, Union, Optional
from copy import deepcopy
import numpy as np
import torch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.ppo.ppo_torch_policy import (kl_and_loss_stats, vf_preds_fetches,
                                                   ppo_surrogate_loss, compute_gae_for_sample_batch)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType, TrainerConfigDict, AgentID

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

VIRTUAL_OBS = 'virtual_obs'

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
        emergency_position = sample_batch[VIRTUAL_OBS][..., 20:22]
    except KeyError:
        emergency_position = sample_batch[SampleBatch.OBS][..., 20:22]
    emergency_states = sample_batch[SampleBatch.OBS][..., 154:190].reshape(-1, 9, 4)
    intrinsic = calculate_intrinsic(agents_position, emergency_position, emergency_states)
    # intrinsic[torch.mean(distance_between_agents < 0.1) > 0] *= 1.5
    sample_batch['original_rewards'] = deepcopy(sample_batch[SampleBatch.REWARDS])
    sample_batch['intrinsic_rewards'] = intrinsic
    if isinstance(sample_batch[SampleBatch.REWARDS], torch.Tensor):
        sample_batch[SampleBatch.REWARDS] += intrinsic
    else:
        sample_batch[SampleBatch.REWARDS] += intrinsic.cpu().numpy()

    return compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)


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
                train_batch['rewards'][fragment] = (train_batch['original_rewards'][fragment] +
                                                    train_batch['intrinsic_rewards'][fragment])
                # additional complete bonus
                train_batch['rewards'][fragment[-1]] += len(fragment) * 5 / episode_length
                train_batch[VIRTUAL_OBS][fragment] = obs_to_be_relabeled
            # for successful trajectories, the aoi is not 100% correct (smaller than actual), check if needed.
            predictor_inputs.append(obs_to_be_relabeled[0])
            predictor_labels.append(len(fragment) / episode_length)
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


def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    """Attach virtual obs to sample batch for Intrinsic reward calculation."""
    extra_dict = vf_preds_fetches(policy, input_dict, state_batches, model, action_dist)
    try:
        extra_dict['virtual_obs'] = model.last_virtual_obs
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
    for item in ['regress_loss', 'label_count', 'valid_fragment_length', 'relabel_percentage']:
        original_dict[item] = torch.mean(torch.stack(policy.get_tower_stats("mean_" + item)))
    for model in policy.model_gpu_towers:
        if model.last_emergency_mode is not None:
            for i in range(model.n_agents):
                original_dict[f'agent_{i}_mode'] = model.last_emergency_mode[i]
                original_dict[f'agent_{i}_target_x'] = model.last_emergency_target[i][0]
                original_dict[f'agent_{i}_target_y'] = model.last_emergency_target[i][1]
                original_dict[f'agent_{i}_count'] = model.last_emergency_count[i]
                original_dict[f'agent_{i}_wait'] = model.last_wait_time[i]
        break
    return original_dict


def ppo_surrogate_loss_debug(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    # print(train_batch['rewards'].shape)
    # print(train_batch['rewards'][119::120])
    return ppo_surrogate_loss(policy, model, dist_class, train_batch)
