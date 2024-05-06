from typing import Dict, List, Optional, Tuple

from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()
from torch import Tensor
import numpy as np
from torch.nn.functional import softmax


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

    def __init__(self, dim: int, with_softmax=True):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.with_softmax = with_softmax

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[
        Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        if self.with_softmax:
            attn = softmax(score, -1)
        else:
            attn = score
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
        self.keys_input_features = model_config['keys_input_features']
        self.with_softmax = model_config['with_softmax']
        self.keys_fc = SlimFC(
            in_size=self.keys_input_features,
            out_size=self.attention_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.values_fc = SlimFC(
            in_size=self.keys_input_features,
            out_size=self.attention_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.attention = ScaledDotProductAttention(self.attention_dim, self.with_softmax)
        self.output_dim = self.attention_dim

    def forward(self, input_dict: Dict[str, TensorType]) -> (TensorType, List[TensorType]):
        emergency = input_dict['emergency']
        agents_state = input_dict['agents_state']
        mask = input_dict.get('mask')  # Optionally include a mask in the inputs

        key_embedding = self.keys_fc(agents_state.unsqueeze(1))
        value_embedding = self.values_fc(agents_state.unsqueeze(1))

        attn_outputs, attn_output_weights = self.attention(emergency, key_embedding, value_embedding, mask=mask)
        return attn_outputs.mean(axis=1), attn_output_weights.squeeze(-1)


class SlimFC(nn.Module):
    """ Simple Fully Connected Layer """

    def __init__(self, in_size, out_size, initializer=None, activation_fn=None):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)
        # Example initializer function, applying a uniform distribution
        if initializer:
            initializer(self.fc.weight)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.fc(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


def normc_initializer(std=1.0):
    def _initializer(weight):
        nn.init.normal_(weight, mean=0, std=std)

    return _initializer


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention mechanism """

    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads

        assert model_dim % num_heads == 0, "Model dimension must be divisible by number of heads"

    def forward(self, queries, keys, values):
        batch_size = queries.size(0)
        # Split into heads
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim)

        # Scaled Dot-Product Attention
        scores = torch.einsum('bnhd,bmhd->bnhm', queries, keys)
        scores /= self.head_dim ** 0.5
        attn = softmax(scores, dim=-1)
        output = torch.einsum('bnhm,bmhd->bnhd', attn, values)

        # Concatenate heads
        output = output.contiguous().view(batch_size, -1, self.model_dim)
        return output, attn


class SelfAttentionEncoder(nn.Module):
    """
    Self Attention Mechanism with Multiple Layers and Skip Connections
    """

    def __init__(self, model_config):
        super(SelfAttentionEncoder, self).__init__()
        self.attention_dim = model_config['attention_dim']
        self.input_dim = model_config['input_dim']
        self.num_heads = model_config['num_heads']
        self.num_layers = model_config['num_layers']  # Number of layers

        # Fully connected layers for transforming inputs
        self.keys_fc = SlimFC(self.input_dim, self.attention_dim, initializer=normc_initializer(0.01))
        self.values_fc = SlimFC(self.input_dim, self.attention_dim, initializer=normc_initializer(0.01))
        self.query_fc = SlimFC(self.input_dim, self.attention_dim, initializer=normc_initializer(0.01))

        # Stack of Multi-Head Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(self.num_heads, self.attention_dim) for _ in range(self.num_layers)
        ])

    def forward(self, agents_state) -> TensorType:
        # Initial projections
        queries = self.query_fc(agents_state)
        keys = self.keys_fc(agents_state)
        values = self.values_fc(agents_state)

        # Pass through each attention layer with skip connection
        attn_output = queries
        for attn_layer in self.attention_layers:
            new_attn_output, _ = attn_layer(attn_output, keys, values)
            attn_output = attn_output + new_attn_output  # Skip connection

        return attn_output
