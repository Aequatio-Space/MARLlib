from collections import namedtuple, deque
from random import sample

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

import torch.nn.functional as F

Transition = namedtuple('Transition', ('goal', 'label'))


class GoalStorage(object):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.position = 0

    def put(self, state, label):
        self.memory.append(Transition(state, label))
        self.position = (self.position + 1) % self.capacity

    def putAll(self, states, labels):
        """
        Put a list of states and labels into the memory.
        :param states: list of states
        :param labels: list of labels
        """
        # create a list of Transitions
        transitions = [Transition(state, label) for state, label in zip(states, labels)]
        # use extend of deque
        self.memory.extend(transitions)

    def sample(self, batch_size):
        transitions = sample(self.memory, batch_size)
        return Transition(*(zip(*transitions)))

    def __len__(self):
        return len(self.memory)

    def len(self):
        return len(self.memory)


class GoalDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=16):
        super(GoalDiscriminator, self).__init__()
        self.pre_embed = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input_features):
        embed = self.pre_embed(input_features)
        query = F.relu(self.linear1(embed))
        x = self.linear2(query)
        return x, query


class AttentiveEncoder(nn.Module):
    """
    Attentive Encoder for goal discriminator
    """

    def __init__(self, num_classes=16, embedding_dim=32, hidden_dim=256, use_lstm=False):
        """
        :param num_classes: number of classes
        :param embedding_dim: embedding dimension
        :param hidden_dim: hidden dimension
        :param use_lstm: whether to use lstm
        """
        super(AttentiveEncoder, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        half_hidden = hidden_dim // 2
        self.use_lstm = use_lstm
        self.target_att_linear = nn.Linear(embedding_dim, half_hidden)
        self.lstm = nn.LSTMCell(hidden_dim, half_hidden)
        self.layer_query = nn.Linear(half_hidden, half_hidden)
        self.layer_key = nn.Linear(half_hidden, half_hidden)
        self.before_attn = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, half_hidden)
        # Note, Policy and value will be feed later.

    def forward(self, img_feat, instruction_idx, _hx, _cx, query):
        word_embedding = self.embedding(instruction_idx)
        word_embedding = word_embedding.view(word_embedding.size(0), -1)
        word_embedding = self.target_att_linear(word_embedding)
        gated_att = torch.sigmoid(word_embedding)  # gated_att

        # apply gated attention
        gated_fusion = torch.mul(img_feat, gated_att)
        if self.use_lstm:
            lstm_input = torch.cat([gated_fusion, gated_att], 1)
            if _hx is None and _cx is None:
                hx, cx = self.lstm(lstm_input)
            else:
                hx, cx = self.lstm(lstm_input, (_hx, _cx))
            mlp_input = torch.cat([gated_fusion, hx], 1)
        else:
            mlp_input = gated_fusion
            hx, cx = _hx, _cx

        mlp_attn = F.relu(self.before_attn(mlp_input))
        key, value = torch.chunk(mlp_attn, 2, dim=1)

        w_q = F.relu(self.layer_query(query))
        w_k = F.relu(self.layer_key(key))
        u_t = torch.tanh(w_q + w_k)
        attention_vector = torch.mul(u_t, value)

        if self.use_lstm:
            attn_weight = torch.cat([attention_vector, hx], 1)
        else:
            attn_weight = attention_vector

        attn_weight = F.relu(self.attn(attn_weight))

        return attn_weight, hx, cx
