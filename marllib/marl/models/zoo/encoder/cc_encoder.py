# MIT License
import logging

import math

from gym import spaces
from functools import reduce
from marllib.marl.algos.utils.centralized_Q import get_dim
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

from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, List
from marllib.marl.models.zoo.encoder.base_encoder import MixInputEncoderMixin

from warp_drive.utils.constants import Constants

torch, nn = try_import_torch()


class CentralizedEncoder(nn.Module, MixInputEncoderMixin):
    """Generic fully connected network."""

    def __init__(
            self,
            model_config,
            obs_space
    ):
        nn.Module.__init__(self)

        # decide the model arch
        self.custom_config = model_config["custom_model_config"]
        self.activation = model_config.get("fcnet_activation")
        self.num_agents = self.custom_config["num_agents"]
        self.encoder_layer_dim = []
        # self.state_dim = None
        # self.state_dim_last = None
        # encoder
        if ("fc_layer" in self.custom_config["model_arch_args"] and
                "conv_layer" in self.custom_config["model_arch_args"]):
            self.mix_input = True
            self.encoder = nn.ModuleDict()
            # reconstruct obs_space so that it contains only 1D space
            vec_obs_space = spaces.Dict({key: obs_space[key][Constants.VECTOR_STATE] for key in obs_space})
            image_obs_space = spaces.Dict({key: obs_space[key][Constants.IMAGE_STATE] for key in obs_space})
            fc_layers, dim1 = self.construct_fc_layer(vec_obs_space)
            self.encoder['fc'] = nn.Sequential(*fc_layers)
            cnn_layers, dim2 = self.construct_cnn_layers(image_obs_space)
            self.encoder['cnn'] = nn.Sequential(*cnn_layers)
            input_dim = dim1 + dim2
        else:
            self.mix_input = False
            if "fc_layer" in self.custom_config["model_arch_args"]:
                layers, input_dim = self.construct_fc_layer(obs_space)
            elif "conv_layer" in self.custom_config["model_arch_args"]:
                layers, input_dim = self.construct_cnn_layers(obs_space)
            else:
                raise ValueError("fc_layer/conv layer not in model arch args")
            self.encoder = nn.Sequential(*layers)

        self.output_dim = input_dim

        my_space = self.custom_config['space_obs']
        if self.mix_input:
            self.obs_vec_dim = get_dim(my_space['obs'][Constants.VECTOR_STATE].shape)
            self.obs_image_dim = get_dim(my_space['obs'][Constants.IMAGE_STATE].shape)
            self.obs_dim = self.obs_vec_dim + self.obs_image_dim
        else:
            self.obs_dim = get_dim(my_space['obs'].shape)
        # repeat for state
        try:
            if self.mix_input:
                self.state_vec_dim = get_dim(my_space['state'][Constants.VECTOR_STATE].shape)
                self.state_image_dim = get_dim(my_space['state'][Constants.IMAGE_STATE].shape)
                self.state_dim = self.state_vec_dim + self.state_image_dim
            else:
                self.state_dim = get_dim(my_space['state'].shape)
        except KeyError:
            logging.debug("No State Space Detected")
            self.state_vec_dim = 0
            self.state_image_dim = 0
            self.state_dim = 0

    def construct_cnn_layers(self, obs_space):
        layers = []
        if "state" not in obs_space.spaces:
            # self.state_dim = obs_space["obs"].shape
            # self.state_dim_last = self.state_dim[-1]
            input_dim = obs_space['obs'].shape[0]
        else:
            # self.state_dim = obs_space["state"].shape
            # self.state_dim_last = obs_space["state"].shape[2] + obs_space["obs"].shape[2]
            input_dim = obs_space['state'].shape[0] + obs_space['obs'].shape[0]
        i = 0
        for i in range(self.custom_config["model_arch_args"]["conv_layer"]):
            layers.append(
                SlimConv2d(
                    in_channels=input_dim,
                    out_channels=self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)],
                    kernel=self.custom_config["model_arch_args"]["kernel_size_layer_{}".format(i)],
                    stride=self.custom_config["model_arch_args"]["stride_layer_{}".format(i)],
                    padding=self.custom_config["model_arch_args"]["padding_layer_{}".format(i)],
                    activation_fn=self.activation
                )
            )
            pool_f = nn.MaxPool2d(kernel_size=self.custom_config["model_arch_args"]["pool_size_layer_{}".format(i)])
            layers.append(pool_f)
            input_dim = self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)]
        layers.append(nn.Flatten())
        linear_input = self.cnn_to_fc_convert(i)
        layers.append(nn.Linear(linear_input,
                                self.custom_config["model_arch_args"][f"out_channel_layer_{i}"]))
        return layers, input_dim

    def construct_fc_layer(self, obs_space):
        fc_layers = []
        encoder_layer_dim = self.get_encoder_layer()
        self.encoder_layer_dim = encoder_layer_dim
        if "state" not in obs_space.spaces:
            input_dim = self.num_agents * obs_space['obs'].shape[0]
        else:
            input_dim = obs_space['state'].shape[0] + obs_space['obs'].shape[0]
        for out_dim in self.encoder_layer_dim:
            fc_layers.append(
                SlimFC(in_size=input_dim,
                       out_size=out_dim,
                       initializer=normc_initializer(1.0),
                       activation_fn=self.activation))
            input_dim = out_dim
        return fc_layers, input_dim

    def cnn_forward(self, cnn_network: nn.Module, inputs):
        """
        Forward pass for CNN
        """
        # x = inputs.reshape(-1, self.state_dim[0], self.state_dim[1], self.state_dim_last)
        return cnn_network(inputs)

    def forward(self, inputs) -> (TensorType, List[TensorType]):
        if self.mix_input:
            vector_input = torch.cat((inputs[..., :self.obs_vec_dim],
                                      inputs[..., self.obs_dim:self.obs_dim + self.state_vec_dim]), dim=-1)
            if len(inputs.shape) > 2:
                flatten_batch_size = math.prod(inputs.shape[:2])
            else:
                flatten_batch_size = inputs.shape[0]
            image_input = torch.cat(
                (inputs[..., self.obs_vec_dim:self.obs_dim].reshape((flatten_batch_size, -1, 10, 10)),
                 inputs[..., self.obs_dim + self.state_vec_dim:].reshape((flatten_batch_size, -1, 10, 10))), dim=1)
            output = torch.cat([self.encoder['fc'](vector_input).reshape(flatten_batch_size, -1),
                                self.cnn_forward(self.encoder['cnn'], image_input)], dim=-1)
        else:
            # Compute the unmasked logits.
            if "conv_layer" in self.custom_config["model_arch_args"]:
                # x = inputs.reshape(-1, self.state_dim[0], self.state_dim[1], self.state_dim_last)
                # x = self.encoder(x.permute(0, 3, 1, 2))
                output = self.encoder(inputs)
            else:
                inputs = inputs.reshape(inputs.shape[0], -1)
                output = self.encoder(inputs)

        return output
