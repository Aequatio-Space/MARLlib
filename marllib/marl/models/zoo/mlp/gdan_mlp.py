from collections import namedtuple, deque
from random import sample

import numpy as np
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

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
        if isinstance(transitions[0].goal, dict):
            # concatenate tensor in the dict
            list_of_dicts = [t.goal for t in transitions]
            state_dict = {
                key: np.stack([d[key] for d in list_of_dicts])
                for key in list_of_dicts[0].keys()
            }
            labels = [t.label for t in transitions]
            return Transition(state_dict, labels)
        else:
            return Transition(*(zip(*transitions)))

    def __len__(self):
        return len(self.memory)

    def len(self):
        return len(self.memory)


class GoalDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=16):
        super(GoalDiscriminator, self).__init__()
        self.hidden_size = hidden_size
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

    def __init__(self, num_classes=16, embedding_dim=32, img_feat_dim=136,
                 hidden_dim=256, num_outputs=9, use_lstm=False):
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
        self.img_feat_project = nn.Linear(img_feat_dim, half_hidden)
        self.target_att_linear = nn.Linear(embedding_dim, half_hidden)

        if self.use_lstm:
            self.lstm = nn.LSTMCell(hidden_dim, half_hidden)
        else:
            self.lstm = None
            # hidden_dim //= 2
            # half_hidden //= 2

        self.before_attn = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, half_hidden)
        self.layer_query = nn.Linear(half_hidden, half_hidden)
        self.layer_key = nn.Linear(half_hidden, half_hidden)
        self.output_dim = half_hidden

        self.mlp_policy1 = nn.Linear(half_hidden, 128)
        self.mlp_policy2 = nn.Linear(128, 64)
        self.p_branch = nn.Linear(64, num_outputs)

        self.mlp_value1 = nn.Linear(half_hidden, 64)
        self.mlp_value2 = nn.Linear(64, 32)  # 64
        self.vf_branch = nn.Linear(32, 1)

        self.last_cx = self.last_hx = None

        self._features = None
        # Note, Policy and value will be feed later.

    def forward(self, img_feat, instruction_idx, _hx, _cx, query):
        img_feat = self.img_feat_project(img_feat)
        word_embedding = self.embedding(instruction_idx)
        word_embedding = word_embedding.view(word_embedding.size(0), -1)
        word_embedding = self.target_att_linear(word_embedding)
        gated_att = torch.sigmoid(word_embedding)  # gated_att

        # apply gated attention
        gated_fusion = torch.mul(img_feat, gated_att)
        if self.use_lstm:
            lstm_input = torch.cat([gated_fusion, gated_att], 1)
            batch_size = img_feat.size(0)
            if _hx is None and _cx is None:
                hx, cx = self.lstm(lstm_input)
                self.last_hx = torch.zeros_like(hx)
                self.last_cx = torch.zeros_like(cx)
            else:
                # hack for dummy batch passing, should fix.
                if _hx.shape[0] != batch_size:
                    hx, cx = self.lstm(lstm_input)
                    self.last_hx = torch.zeros_like(hx)
                    self.last_cx = torch.zeros_like(cx)
                else:
                    self.last_hx = _hx
                    self.last_cx = _cx
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
        self._features = attn_weight
        return attn_weight, hx, cx

    def get_logits(self, features) -> TensorType:
        assert self._features is not None, "must call forward() first"
        policy_embedding = F.relu(self.mlp_policy1(features))
        policy_embedding = F.relu(self.mlp_policy2(policy_embedding))
        return self.p_branch(policy_embedding)

    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        value_embedding = F.relu(self.mlp_value1(self._features))
        value_embedding = F.relu(self.mlp_value2(value_embedding))
        return torch.reshape(self.vf_branch(value_embedding), [-1])
