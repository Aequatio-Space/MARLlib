from typing import Dict, List

from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


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
        self.emergency_feature = 2
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
        self.attention = nn.MultiheadAttention(self.attention_dim, self.num_heads, batch_first=True)
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
