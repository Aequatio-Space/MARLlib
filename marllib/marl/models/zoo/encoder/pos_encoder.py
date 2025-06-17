import torch.nn as nn
import torch
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(1)

        # Linear transformations
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads
        Q = Q.view(-1, batch_size, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(-1, batch_size, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(-1, batch_size, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention = torch.softmax(scores, dim=-1)

        # Apply attention to V
        output = torch.matmul(attention, V)

        # Reshape and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(-1, batch_size, self.d_model)
        return self.W_o(output)


class SimpleTransformer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with residual connection and layer norm
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class LocPredEncoder(nn.Module):
    def __init__(self, max_pos_value):
        super(LocPredEncoder, self).__init__()
        self.hidden_dim = 64
        self.pos_encoder = nn.Linear(2, self.hidden_dim)
        self.max_pos_value = max_pos_value
        # Add CLS token as a learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Use our custom transformer
        self.transformer = SimpleTransformer(
            d_model=self.hidden_dim,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1
        )

    def get_encoding(self, x, y):
        x /= self.max_pos_value
        y /= self.max_pos_value
        pos = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        encode_pos = self.pos_encoder(pos)
        return encode_pos

    def forward(self, x, y):
        """
        x: [horizon + 1, num_envs, num_agents]
        y: [horizon + 1, num_envs, num_agents]
        """
        x /= self.max_pos_value
        y /= self.max_pos_value
        pos = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        pos_encoding = self.pos_encoder(pos)
        _, num_envs, num_agents, _ = pos_encoding.shape
        # Reshape for transformer: [sequence_length, batch_size, features]
        pos_encoding = rearrange(pos_encoding, 'h e a f -> h (e a) f')

        # Add CLS token to the sequence
        batch_size = pos_encoding.size(1)
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)
        pos_encoding = torch.cat([cls_tokens, pos_encoding], dim=0)

        # Apply transformer
        transformer_output = self.transformer(pos_encoding)

        # Split CLS token and sequence
        cls_output = transformer_output[0]  # [batch_size, hidden_dim]
        # seq_output = transformer_output[1:]  # [seq_len, batch_size, hidden_dim]

        cls_output = rearrange(cls_output, '(e a) f -> e a f', e=num_envs, a=num_agents)
        # # Reshape sequence output back to original format
        # seq_output = rearrange(seq_output, 'h (e a) f -> (e a) h f', 
        #                      e=pos.size(1), a=pos.size(2))

        return cls_output
