import logging
from ray.rllib.utils.framework import try_import_torch
from gym import spaces
from marllib.marl.models.zoo.encoder.attention_encoder import ScaledDotProductAttention
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.torch_ops import FLOAT_MAX

from warp_drive.utils.constants import Constants

torch, nn = try_import_torch()


class Predictor(nn.Module):
    def __init__(self, input_dim=22, hidden_size=64, output_dim=1):
        super(Predictor, self).__init__()
        self.activation = nn.ReLU
        self.fc = nn.Sequential(
            SlimFC(
                in_size=input_dim,
                out_size=hidden_size,
                initializer=normc_initializer(0.01),
                activation_fn=self.activation),
            SlimFC(
                in_size=hidden_size,
                out_size=hidden_size,
                initializer=normc_initializer(0.01),
                activation_fn=self.activation),
            SlimFC(
                in_size=hidden_size,
                out_size=output_dim,
                initializer=normc_initializer(0.01),
                activation_fn=None),
            # activation_fn=None),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.fc(x)


class AgentSelector(nn.Module):
    def __init__(self, model_config: dict, obs_space, num_agents):
        super(AgentSelector, self).__init__()
        self.num_agents = num_agents
        self.activation = nn.ReLU
        self.custom_config = model_config['custom_model_config']
        self.model_arch_args = self.custom_config['model_arch_args']
        self.use_2d_state = 'conv_layer' in self.model_arch_args
        self.use_attention = 'attention' in self.model_arch_args['core_arch']
        self.emergency_feature_dim = self.custom_config['emergency_feature_dim']
        self.last_weight_matrix = None
        if self.use_attention:
            # does not support 1d state
            # agent_encoder_arch = {}
            # agent_encoder_arch['emergency_queue_length'] = self.custom_config['emergency_queue_length']
            # agent_encoder_arch['agents_state_dim'] = self.custom_config['status_dim']
            # agent_encoder_arch['num_heads'] = self.model_arch_args['num_heads']
            # agent_encoder_arch['attention_dim'] = self.model_arch_args['attention_dim'] * self.model_arch_args[
            #     'num_heads']
            # agent_encoder_arch['emergency_feature'] = self.emergency_feature_dim
            # agent_encoder_arch['with_softmax'] = True
            self.attention = ScaledDotProductAttention(self.model_arch_args['attention_dim'], with_softmax=True)
            encoder_obs_space = spaces.Dict({
                "obs": spaces.Dict({
                    Constants.VECTOR_STATE: spaces.Box(low=-float('inf'), high=float('inf'),
                                                       shape=(self.custom_config['status_dim'],)),
                    Constants.IMAGE_STATE: spaces.Box(low=-float('inf'), high=float('inf'),
                                                      shape=(1, *obs_space['obs'][Constants.IMAGE_STATE].shape[1:])),
                })
            })
        else:
            encoder_obs_space = obs_space
        self.encoder = BaseEncoder(model_config, encoder_obs_space)
        # self.dropout = nn.Dropout(p=0.3)
        self.query_fc = SlimFC(
            in_size=self.emergency_feature_dim,
            out_size=self.encoder.output_dim,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.out_branch = SlimFC(
            in_size=self.encoder.output_dim,
            out_size=num_agents,
            initializer=normc_initializer(0.01),
            activation_fn=None)

    def forward(self, input_obs, invalid_mask=None, grid=None):
        if self.use_2d_state:
            if self.use_attention:
                agents_feature, emergency = (input_obs[..., :-self.emergency_feature_dim],
                                             input_obs[..., -self.emergency_feature_dim:])
                logging.debug("Agent State Shape: {}".format(agents_feature.shape))
                logging.debug("Grid Shape: {}".format(grid.shape))
                batch_size = agents_feature.shape[0]
                input_dict = {
                    "obs": agents_feature.reshape(self.num_agents * batch_size, self.custom_config['status_dim']),
                    "grid": grid.reshape(self.num_agents * batch_size, 1, *grid.shape[2:])
                }
                embedding = self.encoder(input_dict).reshape(batch_size, self.num_agents, -1)
                query_embedding = self.query_fc(emergency).reshape(batch_size, 1, -1)
                embedding, weights_matrix = self.attention(query_embedding, embedding, embedding)
                embedding, weights_matrix = embedding.squeeze(1), weights_matrix.squeeze(1)
                self.last_weight_matrix = weights_matrix
            else:
                input_dict = {
                    "obs": input_obs,
                    "grid": grid
                }
                embedding = self.encoder(input_dict)

        else:
            if self.use_attention:
                agents_feature, emergency = (input_obs[..., :-self.emergency_feature_dim],
                                             input_obs[..., -self.emergency_feature_dim:])
                agents_feature = agents_feature.reshape(-1, self.num_agents, self.custom_config['status_dim'])
                embedding = self.encoder(agents_feature)
                embedding = self.attention(embedding, emergency)
            else:
                embedding = self.encoder(input_obs)
        raw_output = self.out_branch(embedding)
        if invalid_mask is not None:
            raw_output -= invalid_mask * (FLOAT_MAX / 2)
        action_scores = torch.nn.functional.softmax(raw_output, dim=1)
        return action_scores


class RandomSelector(nn.Module):
    def __init__(self, num_agents, num_actions):
        super(RandomSelector, self).__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions

    def forward(self, input_obs):
        return torch.rand((input_obs.shape[0]))


class GreedySelector(nn.Module):
    def __init__(self, num_agents, num_actions):
        super(GreedySelector, self).__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions

    def forward(self, input_obs):
        agent_positions = input_obs[:, self.num_agents + 2:self.num_agents + 4]
        target_positions = input_obs[:, (self.num_agents + 4) + 4 * (self.num_agents - 1):]
        distances = torch.norm(agent_positions - target_positions, dim=1)
        return distances


class GreedyAgentSelector(nn.Module):
    def __init__(self, input_dim, num_agents):
        super(GreedyAgentSelector, self).__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim

    def forward(self, input_obs, invalid_mask):
        agent_positions = input_obs[:, :-2].reshape(-1, self.num_agents, 3)[..., :2]
        target_positions = input_obs[:, -2:].repeat(self.num_agents, 1).reshape(-1, self.num_agents, 2)
        distances = torch.norm(agent_positions - target_positions, dim=-1)
        distances += (FLOAT_MAX - 5) * invalid_mask
        return torch.nn.functional.one_hot(torch.argmin(distances, dim=-1), self.num_agents)


class RandomAgentSelector(nn.Module):
    def __init__(self, input_dim, num_agents):
        super(RandomAgentSelector, self).__init__()
        self.num_agents = num_agents
        self.input_dim = input_dim

    def forward(self, input_obs):
        return torch.rand((input_obs.shape[0], self.num_agents))
