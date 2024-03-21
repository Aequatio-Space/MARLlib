from typing import Dict, List, Optional, Tuple

from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()
from torch import Tensor
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[
        Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class CrowdSimAttention(nn.Module):
    """
    Attention Mechanism designed to select important emergency
    """

    def __init__(
            self,
            model_config,
    ):
        nn.Module.__init__(self)
        self.attention_dim = model_config['attention_dim']
        self.emergency_queue_length = model_config['emergency_queue_length']
        self.agents_state_dim = model_config['agents_state_dim']
        self.emergency_feature = model_config['emergency_feature']
        self.num_heads = model_config['num_heads']
        self.keys_fc = SlimFC(
            in_size=self.emergency_feature,
            out_size=self.attention_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.values_fc = SlimFC(
            in_size=self.emergency_feature,
            out_size=self.attention_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.query_fc = SlimFC(
            in_size=self.agents_state_dim,
            out_size=self.attention_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.attention = ScaledDotProductAttention(self.attention_dim)
        self.output_dim = self.attention_dim

    def forward(self, input_dict: Dict[str, TensorType]) -> (TensorType, List[TensorType]):
        agents_state, emergency = tuple(input_dict.values())
        agents_embedding = self.query_fc(agents_state.unsqueeze(1))
        emergency = emergency.reshape(agents_state.shape[0], self.emergency_queue_length, self.emergency_feature)
        key_embedding = self.keys_fc(emergency)
        # change embedding dim.
        value_embedding = self.values_fc(emergency)
        attn_outputs, attn_output_weights = self.attention(agents_embedding, key_embedding, value_embedding)
        return attn_outputs.squeeze(1), attn_output_weights.squeeze(1)
