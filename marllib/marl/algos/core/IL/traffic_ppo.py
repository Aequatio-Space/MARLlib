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
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType, TrainerConfigDict, AgentID
from ray.util.annotations import DeveloperAPI

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

VIRTUAL_OBS = 'virtual_obs'

# dirty hack, should be removed later
episode_length = 120


def find_ascending_sequences(arr_tensor: torch.Tensor, min_length=5) -> list[torch.Tensor]:
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

    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        # Input dict is provided to us automatically via the Model's
        # requirements. It's a single-timestep (last one in trajectory)
        # input_dict.
        # Create an input dict according to the Model's requirements.
        input_dict = sample_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last")
        last_r = policy._value(**input_dict)

    # postprocess of rewards
    if other_agent_batches is not None:
        num_agents = len(other_agent_batches) + 1
    else:
        num_agents = 1

    agents_position = sample_batch[SampleBatch.OBS][..., num_agents + 2: num_agents + 4]
    try:
        emergency_position = sample_batch[VIRTUAL_OBS][..., 20:22]
    except KeyError:
        emergency_position = sample_batch[SampleBatch.OBS][..., 20:22]
    emergency_states = sample_batch[SampleBatch.OBS][..., 154:190].reshape(-1, 9, 4)
    intrinsic = calculate_intrinsic(agents_position, emergency_position, emergency_states)
    sample_batch['original_rewards'] = deepcopy(sample_batch[SampleBatch.REWARDS])
    sample_batch['intrinsic_rewards'] = intrinsic
    if isinstance(sample_batch[SampleBatch.REWARDS], torch.Tensor):
        sample_batch[SampleBatch.REWARDS] += intrinsic
    else:
        sample_batch[SampleBatch.REWARDS] += intrinsic.cpu().numpy()

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True))

    return batch


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


def ppo_surrogate_loss(
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
    virtual_obs = train_batch[VIRTUAL_OBS]
    relabel_targets = find_ascending_sequences(train_batch['intrinsic_rewards'])
    num_agents = 4
    predictor_inputs = []
    predictor_labels = []
    train_batch_device = train_batch['intrinsic_rewards'].device
    for fragment in relabel_targets:
        obs_to_be_relabeled = virtual_obs[fragment]
        agent_position = obs_to_be_relabeled[..., num_agents + 2: num_agents + 4].to(model.device)
        emergency_position = obs_to_be_relabeled[..., 20:22].to(model.device)
        # ((250/7910)^2+(250/6960)^2)^0.5 = 0.048, the coverage distance example
        if torch.norm(emergency_position[-1] - agent_position[-1]) > 0.05:
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
    if len(predictor_inputs) > 0:
        predictor_inputs = torch.stack(predictor_inputs).to(train_batch_device)
        predictor_labels = torch.tensor(predictor_labels).float().to(train_batch_device)
        regress_result = model.selector(predictor_inputs).squeeze(-1)
        # L2 Loss
        regress_loss = torch.nn.MSELoss()(regress_result, predictor_labels).to(model.device)
    else:
        regress_loss = torch.tensor(0.0).to(model.device)

    mean_regress_loss = torch.mean(regress_loss)
    model.train()
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
                                  model)

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
        train_batch[SampleBatch.ACTION_LOGP])
    action_kl = prev_action_dist.kl(curr_action_dist).to(model.device)
    mean_kl_loss = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy().to(model.device)
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"],
            1 + policy.config["clip_param"])).to(model.device)
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    # Compute a value function loss.
    if policy.config["use_critic"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
        value_fn_out = model.value_function()
        vf_loss1 = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        vf_loss2 = torch.pow(
            vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2).to(model.device)
        mean_vf_loss = reduce_mean_valid(vf_loss)
    # Ignore the value function.
    else:
        vf_loss = mean_vf_loss = 0.0

    total_loss = reduce_mean_valid(-surrogate_loss +
                                   policy.kl_coeff * action_kl +
                                   policy.config["vf_loss_coeff"] * vf_loss -
                                   policy.entropy_coeff * curr_entropy + regress_loss)
    model.eval()
    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss
    model.tower_stats['mean_regress_loss'] = mean_regress_loss

    return total_loss


def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    """Attach virtual obs to sample batch for Intrinsic reward calculation."""
    extra_dict = {SampleBatch.VF_PREDS: model.value_function()}
    try:
        extra_dict['virtual_obs'] = model.last_virtual_obs
        # extra_dict['predicted_values'] = model.last_predicted_values
    except AttributeError:
        pass
    return extra_dict
