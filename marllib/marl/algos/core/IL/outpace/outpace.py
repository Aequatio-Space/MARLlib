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

from queue import Queue
from .hgg import TrajectoryPool, MatchSampler


def after_loss_init(policy: Policy, observation_space: gym.spaces.Space,
                    action_space: gym.spaces.Space, config: TrainerConfigDict) -> None:
    # TODO: config is placeholder
    setattr(policy, "hgg_sampler_update_frequency", 20)
    setattr(policy, "hgg_achieved_trajectory_pool", TrajectoryPool(100))
    setattr(policy, "hgg_sampler", MatchSampler(env_name='crowdsim', config=config,
                                                achieved_trajectory_pool=policy.hgg_achieved_trajectory_pool,
                                                **config['hgg_kwargs']['match_sampler_kwargs']
                                                ))
    setattr(policy, "recent_sampled_goals", Queue(policy.hgg_sampler_update_frequency))


def get_policy_class_outpace(config_):
    if config_["framework"] == "torch":
        return OUTPACETorchPolicy


OUTPACETorchPolicy = PPOTorchPolicy.with_updates(
    name="OUTPACETorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    # postprocess_fn=relabel_for_sample_batch,
    # loss_fn=add_auxiliary_loss,
    # extra_action_out_fn=extra_action_out_fn,
    # stats_fn=kl_and_loss_stats_with_regress,
    _after_loss_init=after_loss_init,
)

OUTPACETrainer = WandbPPOTrainer.with_updates(
    name="OUTPACETrainer",
    default_policy=None,
    get_policy_class=get_policy_class_outpace,
)
