# MIT License
import logging
from typing import Union

import math
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
from warp_drive.utils.constants import Constants
from .attention_encoder import CrowdSimAttention
import logging

torch, nn = try_import_torch()

class MixInputEncoderMixin:
    """
    Mix input convert mixin
    Common functions for mix input encoder.
    Do not use separately.
    """
    custom_config = {}

    def get_encoder_layer(self):
        if "encode_layer" in self.custom_config["model_arch_args"]:
            encode_layer = self.custom_config["model_arch_args"]["encode_layer"]
            encoder_layer_dim = encode_layer.split("-")
            encoder_layer_dim = [int(i) for i in encoder_layer_dim]
        else:  # default config
            encoder_layer_dim = []
            for i in range(self.custom_config["model_arch_args"]["fc_layer"]):
                out_dim = self.custom_config["model_arch_args"]["out_dim_fc_{}".format(i)]
                encoder_layer_dim.append(out_dim)
        return encoder_layer_dim

    def cnn_to_fc_convert(self, i):
        out_channels = self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)]
        last_padding = self.custom_config["model_arch_args"]["padding_layer_{}".format(i)]
        last_cnn_kernel = self.custom_config["model_arch_args"]["kernel_size_layer_{}".format(i)]
        last_cnn_stride = self.custom_config["model_arch_args"]["stride_layer_{}".format(i)]
        last_pool_kernel = self.custom_config["model_arch_args"]["pool_size_layer_{}".format(i)]
        feature_length = math.ceil((10 + last_padding * 2 - last_cnn_kernel + 1) / last_cnn_stride) // last_pool_kernel
        linear_input = out_channels * feature_length * feature_length
        return linear_input


class BaseEncoder(nn.Module, MixInputEncoderMixin):
    """Generic fully connected network."""

    def __init__(
            self,
            model_config,
            obs_space
    ):
        nn.Module.__init__(self)

        # decide the model arch
        self.custom_config = model_config["custom_model_config"]
        self.fc_activation = model_config.get("fcnet_activation")
        self.cnn_activation = model_config.get("conv_activation")
        self.mix_input = False
        self.inputs = None
        self.use_fc = "fc_layer" in self.custom_config["model_arch_args"]
        self.use_cnn = "conv_layer" in self.custom_config["model_arch_args"]
        self.use_attention = "attention_layer" in self.custom_config['model_arch_args']
        # encoder
        if (self.use_fc and self.use_cnn) or (self.use_attention and self.use_cnn):
            self.mix_input = True
            self.encoder = nn.ModuleDict()
            if self.use_fc:
                fc_layers, dim1 = self.construct_fc_layer(obs_space['obs'][Constants.VECTOR_STATE])
                self.encoder['fc'] = nn.Sequential(*fc_layers)
            if self.use_attention:
                self.encoder['attention'] = CrowdSimAttention(self.custom_config['model_arch_args'])
            cnn_layers, dim2 = self.construct_cnn_layers(obs_space['obs'][Constants.IMAGE_STATE])
            self.encoder['cnn'] = nn.Sequential(*cnn_layers)
            self.output_dim = dim1 + dim2
            self.dims = [dim1, dim2]
        else:
            if self.use_fc:
                layers, input_dim = self.construct_fc_layer(obs_space['obs'])
            elif self.use_cnn:
                layers, input_dim = self.construct_cnn_layers(obs_space['obs'])
            else:
                raise ValueError("fc_layer/conv layer not in model arch args")
            self.encoder = nn.Sequential(*layers)
            self.output_dim = input_dim  # record
            self.dims = [input_dim]
        logging.debug(f"Encoder Configuration: {self.encoder}")

    def construct_cnn_layers(self, obs_space):
        """
        Construct CNN layers
        """
        input_dim = obs_space.shape[0]
        cnn_layers = []
        i = 0
        for i in range(self.custom_config["model_arch_args"]["conv_layer"]):
            cnn_layers.append(
                SlimConv2d(
                    in_channels=input_dim,
                    out_channels=self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)],
                    kernel=self.custom_config["model_arch_args"]["kernel_size_layer_{}".format(i)],
                    stride=self.custom_config["model_arch_args"]["stride_layer_{}".format(i)],
                    padding=self.custom_config["model_arch_args"]["padding_layer_{}".format(i)],
                    activation_fn=self.cnn_activation,
                )
            )
            pool_f = nn.MaxPool2d(kernel_size=self.custom_config["model_arch_args"]["pool_size_layer_{}".format(i)])
            cnn_layers.append(pool_f)
            input_dim = self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)]
        cnn_layers.append(nn.Flatten())
        linear_input = self.cnn_to_fc_convert(i)
        cnn_layers.append(nn.Linear(linear_input,
                                    self.custom_config["model_arch_args"][f"out_channel_layer_{i}"]))
        return cnn_layers, input_dim

    def construct_fc_layer(self, obs_space):
        """
        Construct (Multiple) FC layers
        """
        encoder_layer_dim = self.get_encoder_layer()
        # self.encoder_layer_dim = encoder_layer_dim
        input_dim = obs_space.shape[0]
        fc_layers = []
        for out_dim in encoder_layer_dim:
            fc_layers.append(
                SlimFC(in_size=input_dim,
                       out_size=out_dim,
                       initializer=normc_initializer(1.0),
                       activation_fn=self.fc_activation))
            input_dim = out_dim
        return fc_layers, input_dim

    def forward(self, inputs: Union[TensorType, dict]) -> (TensorType, List[TensorType]):
        if self.mix_input:
            state, grid = tuple(inputs.values())
            # logging.debug(f"encoder input shape:{state.shape},{grid.shape}")
            if self.use_fc:
                output = torch.cat([self.encoder['fc'](state),
                                self.cnn_forward(self.encoder['cnn'], grid)], dim=-1)
            elif self.use_attention:
                output = torch.cat([self.encoder['attention'](state),
                                    self.cnn_forward(self.encoder['cnn'], grid)], dim=-1)

        else:
            # Compute the unmasked logits.
            # logging.debug(f"encoder input shape:{inputs.shape}")
            if "conv_layer" in self.custom_config["model_arch_args"]:
                output = self.cnn_forward(self.encoder, inputs)
            else:
                self.inputs = inputs.reshape(inputs.shape[0], -1)
                output = self.encoder(inputs)
        # logging.debug(f"encoder output shape:{output.shape}")
        return output

    def cnn_forward(self, cnn_network: nn.Module, inputs):
        """
        CNN forward, for GRU and LSTM the forward logic is slightly different.
        """
        B = inputs.shape[0]
        if self.custom_config["model_arch_args"]["core_arch"] in ["gru", "lstm"] and len(inputs.shape) > 4:
            L = inputs.shape[1]
            inputs = inputs.reshape((-1,) + inputs.shape[2:])
            # inputs.reshape(B*L, )
            x = cnn_network(inputs)
            x = torch.mean(x, (2, 3))
            output = x.reshape((B, L, -1))
        else:
            logging.debug(f"cnn encoder input shape:{inputs.shape}")
            output = cnn_network(inputs)  # Does not understand why the channel dim is put at the last dim
            # logging.debug(f"cnn encoder output shape:{x.shape}")
            # output = torch.mean(x, (2, 3))
            # logging.debug(f"cnn encoder second output shape:{output.shape}")
        return output


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation_fn):
        super(ResNetBlock, self).__init__()
        self.conv1 = SlimConv2d(in_channels, out_channels, kernel_size, stride, padding, activation_fn)
        self.conv2 = SlimConv2d(out_channels, out_channels, kernel_size, stride, padding, activation_fn)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class ResNetCNNEncoder(nn.Module):

    def __init__(
            self,
            model_config,
            obs_space
    ):
        nn.Module.__init__(self)

        # decide the model arch
        self.custom_config = model_config["custom_model_config"]
        self.cnn_activation = model_config.get("conv_activation")
        # set number of ResNet blocks according to model_config
        self.num_blocks = self.custom_config["model_arch_args"]["num_blocks"]
        # construct the encoder
        self.main_encoder = nn.Sequential(
            *[ResNetBlock(
                in_channels=self.custom_config["model_arch_args"]["out_channel_layer_0"],
                out_channels=self.custom_config["model_arch_args"]["out_channel_layer_0"],
                kernel_size=self.custom_config["model_arch_args"]["kernel_size_layer_0"],
                stride=self.custom_config["model_arch_args"]["stride_layer_0"],
                padding=self.custom_config["model_arch_args"]["padding_layer_0"],
                activation_fn=self.cnn_activation
            ) for _ in range(self.num_blocks - 1)])
        # flatten the input and convert to FC
        self.to_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.custom_config["model_arch_args"]["out_channel_layer_0"] * 10 * 10,
                      self.custom_config["model_arch_args"]["out_channel_layer_0"])
        )

    def forward(self, inputs: Union[TensorType, dict]) -> (TensorType, List[TensorType]):
        if "conv_layer" in self.custom_config["model_arch_args"]:
            output = self.to_fc(self.main_encoder(inputs))
        else:
            raise ValueError("conv_layer not in model arch args")
        return output
