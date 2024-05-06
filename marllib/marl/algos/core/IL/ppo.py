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

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ppo_surrogate_loss
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG

# from marllib.marl.algos.core.IL.trafficppo import (relabel_for_sample_batch, sample_batch_with_demonstration,
#                                                    add_regress_loss, save_sample_batch)
###########
### PPO ###
###########


def ppo_surrogate_loss_fix_kl(policy, model, dist_class, train_batch):
    # underflow prevention for iter_policy.kl_coeff
    policy.kl_coeff = max(policy.kl_coeff, policy.config['kl_coeff_min'])
    return ppo_surrogate_loss(policy, model, dist_class, train_batch)


IPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="IPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    loss_fn=ppo_surrogate_loss_fix_kl,
)

def get_policy_class_ppo(config_):
    if config_["framework"] == "torch":
        return IPPOTorchPolicy

IPPOTrainer = WandbPPOTrainer.with_updates(
    name="IPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_ppo,
)