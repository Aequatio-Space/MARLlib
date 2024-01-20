# MIT License

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

from marllib.marl.algos.wandb_trainers import WandbPPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from marllib.marl.algos.core.IL.traffic_ppo import (
    compute_gae_and_intrinsic_for_sample_batch, add_regress_loss,
    extra_action_out_fn, kl_and_loss_stats_with_regress)

###########
### PPO ###
###########


IPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="IPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    # postprocess_fn=compute_gae_and_intrinsic_for_sample_batch,
    # loss_fn=add_regress_loss,
    # extra_action_out_fn=extra_action_out_fn,
    # stats_fn=kl_and_loss_stats_with_regress,
)

TrafficPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="TrafficPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
)

def get_policy_class_ppo(config_):
    if config_["framework"] == "torch":
        return IPPOTorchPolicy


def get_policy_class_traffic_ppo(config_):
    if config_["framework"] == "torch":
        return TrafficPPOTorchPolicy


IPPOTrainer = WandbPPOTrainer.with_updates(
    name="IPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_ppo,
)

TrafficPPOTrainer = WandbPPOTrainer.with_updates(
    name="TrafficPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_traffic_ppo,
)
